from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_runtime_lifecycle.py"
SPEC = importlib.util.spec_from_file_location(
    "runtime_lifecycle_qualification_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _manifest(run_id: str, results: dict[str, dict[str, str]]) -> dict[str, object]:
    return {
        "schema_version": qualification.RAW_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_contract_sha256": "0" * 64,
        "results": results,
    }


def test_trial_progress_counts_only_atomically_committed_manifest_entries(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    results = raw / "results"
    results.mkdir(parents=True)
    (results / "committed.json").write_text("{}\n", encoding="utf-8")
    (results / "orphan.json").write_text("{}\n", encoding="utf-8")
    manifest = _manifest(
        "trial-interruption",
        {
            "snapshot/backend": {
                "path": "committed.json",
                "planned_trial_id": "snapshot/backend",
                "sha256": "1" * 64,
            }
        },
    )
    path = raw / "raw_result_manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    assert qualification._committed_trial_progress(
        path, expected_run_id="trial-interruption"
    ) == 1


def test_interruption_state_freezes_orphan_temporary_and_inventory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    raw = run_root / "raw"
    results = raw / "results"
    results.mkdir(parents=True)
    committed = results / "committed.json"
    committed.write_text("committed\n", encoding="utf-8")
    (results / "orphan.json").write_text("orphan\n", encoding="utf-8")
    (run_root / ".payload.staging-dead").mkdir()
    manifest = _manifest(
        "trial-interruption",
        {
            "snapshot/backend": {
                "path": committed.name,
                "planned_trial_id": "snapshot/backend",
                "sha256": "2" * 64,
            }
        },
    )
    (raw / "raw_result_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    layout = SimpleNamespace(
        run_id="trial-interruption", run_root=run_root, raw_results=raw
    )

    state = qualification._trial_interruption_state(layout)

    assert state["committed_trial_count"] == 1
    assert state["orphan_result_files"] == ["orphan.json"]
    assert state["orphan_result_count"] == 1
    assert state["temporary_entry_count"] == 1
    assert set(state["result_file_inventory"]) == {"committed.json", "orphan.json"}


def test_execution_counts_come_from_explicit_per_id_evidence() -> None:
    records = qualification._planned_execution_records(
        planned_ids=["a", "b"],
        initial_executed_ids=["a"],
        resume_executed_ids=["a", "b", "b"],
    )

    assert records == [
        {
            "planned_id": "a",
            "initial_execution_count": 1,
            "resume_execution_count": 1,
            "final_execution_count": 2,
        },
        {
            "planned_id": "b",
            "initial_execution_count": 0,
            "resume_execution_count": 2,
            "final_execution_count": 2,
        },
    ]
    assert qualification._valid_reexecution_count(records) == 1
    assert qualification._execution_duplicate_count(records) == 2


def test_empty_worker_source_log_is_strict_zero_evidence(tmp_path: Path) -> None:
    path = tmp_path / "source.ndjson"
    path.touch()

    report = qualification._source_access_log_report(path)

    assert report["source_repository_open_event_count"] == 0
    assert report["SOURCE_ACCESS_LOG_PASS"] is True
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="truncated"):
        qualification._source_access_log_report(path)


def _write_tar(tar_path: Path, root_name: str, files: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w:gz") as archive:
        root = tarfile.TarInfo(root_name)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)
        for relative, payload in files.items():
            info = tarfile.TarInfo(f"{root_name}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))


def test_live_archive_tree_is_bound_to_tar_and_rejects_extra(tmp_path: Path) -> None:
    live = tmp_path / "archive"
    live.mkdir(mode=0o755)
    (live / "evidence.json").write_bytes(b"fixed\n")
    (live / "evidence.json").chmod(0o644)
    tar_path = tmp_path / "archive.tar.gz"
    _write_tar(tar_path, live.name, {"evidence.json": b"fixed\n"})

    assert qualification._compare_live_tree_to_tar(live, tar_path)[
        "LIVE_FAILURE_ARCHIVE_TAR_TREE_PASS"
    ] is True
    (live / "extra.txt").write_text("extra", encoding="utf-8")
    report = qualification._compare_live_tree_to_tar(live, tar_path)
    assert report["LIVE_FAILURE_ARCHIVE_TAR_TREE_PASS"] is False
    assert report["extra_files"] == ["extra.txt"]


def test_tar_path_escape_is_rejected(tmp_path: Path) -> None:
    live = tmp_path / "archive"
    live.mkdir()
    tar_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo("archive/../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(RuntimeError, match="unsafe tar member"):
        qualification._compare_live_tree_to_tar(live, tar_path)


def _external_pcl_v3_inventory_or_skip(
    root: Path = qualification.PCL_V3_SOURCE,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    availability = qualification._pcl_v3_external_qualification_availability(
        root, environ=environ
    )
    if availability["candidate_exists"] is not True:
        pytest.skip(availability["unavailable_reason"])
    return qualification._validated_pcl_v3_inventory(root)


def test_fixed_pcl_v3_input_inventory_matches_contract() -> None:
    report = _external_pcl_v3_inventory_or_skip()

    assert report["file_count"] == 18
    assert report["size_bytes"] == 8855746
    assert report["canonical_rows_json_sha256"] == qualification.EXPECTED_PCL_V3_TREE_SHA256
    assert report["pcl_point_to_plane_cli_sha256"] == qualification.EXPECTED_PCL_V3_CLI_SHA256
    assert report["PCL_V3_INPUT_BINDING_PASS"] is True


def _write_pcl_v3_contract_candidate(root: Path) -> None:
    for name in qualification.PCL_V3_DIRECTORIES:
        (root / name).mkdir(parents=True)
    (root / "bin/pcl_point_to_plane_cli").write_bytes(b"cli\n")
    (root / "src/source.cc").write_bytes(b"source\n")
    (root / "tests/case.json").write_bytes(b"{}\n")
    (root / "tools/verify.py").write_bytes(b"pass\n")


def _bind_expected_pcl_v3_contract(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> dict[str, object]:
    report = qualification._pcl_v3_inventory(root)
    monkeypatch.setattr(
        qualification, "EXPECTED_PCL_V3_FILE_COUNT", report["file_count"]
    )
    monkeypatch.setattr(
        qualification, "EXPECTED_PCL_V3_SIZE_BYTES", report["size_bytes"]
    )
    monkeypatch.setattr(
        qualification,
        "EXPECTED_PCL_V3_TREE_SHA256",
        report["canonical_rows_json_sha256"],
    )
    monkeypatch.setattr(
        qualification,
        "EXPECTED_PCL_V3_CLI_SHA256",
        report["pcl_point_to_plane_cli_sha256"],
    )
    return report


def test_missing_pcl_v3_bundle_skips_in_source_only_mode(tmp_path: Path) -> None:
    missing = tmp_path / "missing-pcl-v3"

    with pytest.raises(pytest.skip.Exception) as skipped:
        _external_pcl_v3_inventory_or_skip(missing, environ={})

    reason = str(skipped.value)
    assert str(missing) in reason
    assert (
        "source-only package: external historical qualification bundle unavailable"
        in reason
    )


def test_missing_pcl_v3_bundle_fails_when_external_qualification_is_required(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-pcl-v3"
    environment = {qualification.PCL_V3_EXTERNAL_QUALIFICATION_ENV: "1"}

    with pytest.raises(FileNotFoundError) as failure:
        _external_pcl_v3_inventory_or_skip(missing, environ=environment)

    message = str(failure.value)
    assert f"{qualification.PCL_V3_EXTERNAL_QUALIFICATION_ENV}=1" in message
    assert str(missing) in message
    assert (
        "source-only package: external historical qualification bundle unavailable"
        in message
    )


def test_existing_pcl_v3_bundle_with_wrong_sha_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "pcl-v3"
    _write_pcl_v3_contract_candidate(candidate)
    _bind_expected_pcl_v3_contract(monkeypatch, candidate)
    (candidate / "bin/pcl_point_to_plane_cli").write_bytes(b"bad\n")

    with pytest.raises(RuntimeError, match="package binding mismatch"):
        _external_pcl_v3_inventory_or_skip(candidate, environ={})


def test_existing_pcl_v3_bundle_matching_contract_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "pcl-v3"
    _write_pcl_v3_contract_candidate(candidate)
    expected = _bind_expected_pcl_v3_contract(monkeypatch, candidate)

    report = _external_pcl_v3_inventory_or_skip(candidate, environ={})

    assert report["file_inventory"] == expected["file_inventory"]
    assert report["PCL_V3_INPUT_BINDING_PASS"] is True


def test_empty_pcl_v3_directory_cannot_pass_or_skip(tmp_path: Path) -> None:
    empty = tmp_path / "empty-pcl-v3"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="package directory is missing or unsafe"):
        _external_pcl_v3_inventory_or_skip(empty, environ={})


def test_worker_command_binds_unique_external_source_log(tmp_path: Path) -> None:
    source_log = tmp_path / "worker.source_access.ndjson"
    command = qualification._worker_command(
        repository=ROOT,
        qualification_root=tmp_path / "qualification",
        run_id="run",
        invocation_id="invocation",
        expected_commit="0" * 40,
        expected_branch="branch",
        expected_tag="tag",
        source_access_log=source_log,
        resume=True,
        delay=0.0,
    )

    index = command.index("--source-access-log")
    assert command[index + 1] == str(source_log)
    assert "--resume" in command
