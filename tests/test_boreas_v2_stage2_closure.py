from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization as auth_module
import phase_a_harness.real_data_preparation.boreas_v2_stage2_closure as closure
from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (
    VerifiedStage2Authorization,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier import (
    PAYLOAD_FILES,
    REQUIRED_FILES,
    UNCERTAINTY_FIELDS,
)
from phase_a_harness.real_data_preparation.guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
)
from phase_a_harness.real_data_preparation.io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)


def _authority(tmp_path: Path) -> types.SimpleNamespace:
    paths = {}
    for name in (
        "pair_selection_path",
        "preprocessing_contract_path",
        "stage1_allowlist_path",
        "stage1_manifest_path",
        "storage_manifest_path",
    ):
        path = tmp_path / f"authority-{name}.json"
        atomic_write_json(path, {"name": name})
        paths[name] = path
    md = paths["preprocessing_contract_path"].with_suffix(".md")
    atomic_write_bytes(md, b"# authoritative preprocessing\n")
    return types.SimpleNamespace(**paths)


def _assembled_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, types.SimpleNamespace]:
    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    candidate = tmp_path / "candidate"
    evidence.mkdir()
    runtime.mkdir()
    (runtime / "checkpoints").mkdir()
    authority = _authority(tmp_path)
    authorization = tmp_path / "authorization.json"
    preprocessing_md = tmp_path / "preprocessing.md"
    preprocessing_json = tmp_path / "preprocessing.json"
    for path in (authorization, preprocessing_json):
        atomic_write_json(path, {"path": path.name})
    atomic_write_bytes(preprocessing_md, b"# preprocessing\n")
    for name in PAYLOAD_FILES - closure.GENERATED_PAYLOADS - closure.EXTERNAL_PAYLOADS:
        atomic_write_bytes(evidence / name, f"fixture:{name}\n".encode())

    fake_summary = {
        "answers": [
            {"answer": True, "item": index, "question": f"q{index}"}
            for index in range(1, 31)
        ],
        "conclusion": "fixture",
        "empirical_counts_and_bytes": {},
        "large_artifact_policy": "fixture",
        "readiness": closure._readiness(),
        "resource_capacity_evidence": {},
        "schema": "zprm.boreas.v2.stage2.preparation_summary.v1",
    }
    monkeypatch.setattr(
        closure,
        "_summary",
        lambda **_: (fake_summary, b"# fixture summary\n"),
    )
    closure.assemble_small_closure_candidate(
        candidate_root=candidate,
        evidence_root=evidence,
        authorization_path=authorization,
        preprocessing_contract_json=preprocessing_json,
        preprocessing_contract_markdown=preprocessing_md,
        backend_contract_path=tmp_path / "unused-backend.json",
        reducer_resource_plan_path=runtime
        / "checkpoints/reducer_resource_plan.json",
        test_status_path=tmp_path / "unused-tests.json",
        disk_gate_events=(),
        runtime_root=runtime,
        authority=authority,
    )
    return candidate, runtime, authority


def test_small_candidate_inventory_is_exact_and_never_copies_large_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, runtime, _ = _assembled_candidate(tmp_path, monkeypatch)
    (runtime / "target_maps").mkdir()
    atomic_write_bytes(runtime / "target_maps/large.npy", b"large-runtime-only")
    (runtime / "snapshots").mkdir()
    atomic_write_bytes(runtime / "snapshots/source.npy", b"snapshot-runtime-only")
    assert {path.name for path in candidate.iterdir()} == set(REQUIRED_FILES)
    assert not any(path.suffix == ".npy" for path in candidate.iterdir())
    assert not any("snapshot" in path.name and path.is_dir() for path in candidate.iterdir())


def test_candidate_complete_resume_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, runtime, authority = _assembled_candidate(tmp_path, monkeypatch)
    before = {name: sha256_file(candidate / name) for name in REQUIRED_FILES}
    monkeypatch.setattr(closure, "_verify_generated_documents", lambda *_, **__: None)
    manifest = closure.assemble_small_closure_candidate(
        candidate_root=candidate,
        evidence_root=tmp_path / "evidence",
        authorization_path=tmp_path / "authorization.json",
        preprocessing_contract_json=tmp_path / "preprocessing.json",
        preprocessing_contract_markdown=tmp_path / "preprocessing.md",
        backend_contract_path=tmp_path / "unused-backend.json",
        reducer_resource_plan_path=runtime
        / "checkpoints/reducer_resource_plan.json",
        test_status_path=tmp_path / "unused-tests.json",
        disk_gate_events=(),
        runtime_root=runtime,
        authority=authority,
    )
    assert manifest["manifest_root_sha256"]
    assert before == {name: sha256_file(candidate / name) for name in REQUIRED_FILES}


def test_publish_calls_candidate_staging_destination_and_reuses_exact_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, runtime, authority = _assembled_candidate(tmp_path, monkeypatch)
    frozen_parent = tmp_path / "frozen_assets"
    frozen_parent.mkdir()
    destination = frozen_parent / "closure"
    monkeypatch.setattr(closure, "_verify_generated_documents", lambda *_, **__: None)
    calls: list[Path] = []

    def verifier(*, root: Path, **_: object) -> dict[str, object]:
        calls.append(Path(root))
        return {
            "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS": True,
            "BOREAS_EXTERNAL_V2_STAGE2_READY": True,
            "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": True,
            "manifest_root_sha256": json.loads(
                (Path(root) / "boreas_v2_stage2_frozen_manifest.json").read_text()
            )["manifest_root_sha256"],
        }

    live_calls: list[None] = []
    publication = closure.verify_and_publish_small_closure(
        candidate_root=candidate,
        destination_root=destination,
        runtime_root=runtime,
        authority=authority,
        live_check=lambda: live_calls.append(None),
        verifier=verifier,
        production_mode=False,
    )
    assert [path.name for path in calls] == ["candidate", calls[1].name, "closure"]
    assert ".staging" in calls[1].name
    assert publication.reused_existing_destination is False
    assert len(live_calls) >= len(REQUIRED_FILES) + 4
    calls.clear()
    reused = closure.verify_and_publish_small_closure(
        candidate_root=candidate,
        destination_root=destination,
        runtime_root=runtime,
        authority=authority,
        live_check=lambda: None,
        verifier=verifier,
        production_mode=False,
    )
    assert reused.reused_existing_destination is True
    assert [path.name for path in calls] == ["candidate", "closure"]


def test_failed_staging_verification_never_publishes_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, runtime, authority = _assembled_candidate(tmp_path, monkeypatch)
    parent = tmp_path / "frozen_assets"
    parent.mkdir()
    destination = parent / "closure"
    monkeypatch.setattr(closure, "_verify_generated_documents", lambda *_, **__: None)
    count = 0

    def verifier(**_: object) -> dict[str, object]:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("staging rejected")
        return {
            "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS": True,
            "manifest_root_sha256": json.loads(
                (candidate / "boreas_v2_stage2_frozen_manifest.json").read_text()
            )["manifest_root_sha256"],
        }

    with pytest.raises(RuntimeError, match="staging rejected"):
        closure.verify_and_publish_small_closure(
            candidate_root=candidate,
            destination_root=destination,
            runtime_root=runtime,
            authority=authority,
            live_check=lambda: None,
            verifier=verifier,
            production_mode=False,
        )
    assert not destination.exists()


def test_publication_intent_recovers_authenticated_partial_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, runtime, authority = _assembled_candidate(tmp_path, monkeypatch)
    parent = tmp_path / "frozen_assets"
    parent.mkdir()
    destination = parent / "closure"
    monkeypatch.setattr(closure, "_verify_generated_documents", lambda *_, **__: None)
    original_copy = closure._copy_payload
    copied = 0

    def interrupted_copy(source: Path, target: Path) -> None:
        nonlocal copied
        if copied == 3:
            raise RuntimeError("injected publication copy crash")
        original_copy(source, target)
        copied += 1

    def verifier(*, root: Path, **_: object) -> dict[str, object]:
        return {
            "BOREAS_EXTERNAL_V2_STAGE2_VERIFICATION_PASS": True,
            "manifest_root_sha256": json.loads(
                (Path(root) / "boreas_v2_stage2_frozen_manifest.json").read_text()
            )["manifest_root_sha256"],
        }

    monkeypatch.setattr(closure, "_copy_payload", interrupted_copy)
    with pytest.raises(RuntimeError, match="injected publication copy crash"):
        closure.verify_and_publish_small_closure(
            candidate_root=candidate,
            destination_root=destination,
            runtime_root=runtime,
            authority=authority,
            live_check=lambda: None,
            verifier=verifier,
            production_mode=False,
        )
    intent = runtime / "checkpoints/closure_publication_intent.json"
    assert intent.is_file()
    staging = next(parent.glob(".*.staging"))
    assert 0 < len(list(staging.iterdir())) < len(REQUIRED_FILES)
    pending_name = sorted(set(REQUIRED_FILES) - {p.name for p in staging.iterdir()})[0]
    atomic_write_bytes(staging / f".{pending_name}.crashwindow.partial", b"partial")

    monkeypatch.setattr(closure, "_copy_payload", original_copy)
    result = closure.verify_and_publish_small_closure(
        candidate_root=candidate,
        destination_root=destination,
        runtime_root=runtime,
        authority=authority,
        live_check=lambda: None,
        verifier=verifier,
        production_mode=False,
    )
    assert result.reused_existing_destination is False
    assert {entry.name for entry in destination.iterdir()} == set(REQUIRED_FILES)
    assert not intent.exists()


def _capability(
    tmp_path: Path, guard: NoRegistrationGuard
) -> VerifiedStage2Authorization:
    authorization = tmp_path / "authorization.json"
    atomic_write_json(authorization, {})
    return VerifiedStage2Authorization(
        _token=auth_module._CAPABILITY_TOKEN,
        _document_bytes=b"{}",
        repository=tmp_path,
        stage1_data_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        temporary_root=tmp_path / "runtime",
        monitored_disk_path=tmp_path,
        authorization_path=authorization,
        authorization_file_sha256=sha256_file(authorization),
        no_registration_guard=guard,
    )


def test_no_icp_prerequisite_comes_from_live_guard_not_fabricated_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    evidence = runtime / "evidence"
    runtime.mkdir()
    evidence.mkdir()
    contract = tmp_path / "contract.json"
    atomic_write_json(contract, {"schema": "fixture"})
    registration = types.SimpleNamespace(registration_icp=lambda: "forbidden")
    open3d = types.SimpleNamespace(
        pipelines=types.SimpleNamespace(registration=registration)
    )
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    monkeypatch.setattr(
        VerifiedStage2Authorization, "assert_operation_live", lambda self: self
    )
    with NoRegistrationGuard(open3d_module=open3d) as guard:
        capability = _capability(tmp_path, guard)
        with pytest.raises(RegistrationForbiddenError):
            registration.registration_icp()
        with pytest.raises(closure.BoreasV2Stage2ClosureError, match="detected"):
            closure.write_selection_prerequisites(
                evidence_root=evidence,
                preprocessing_contract_path=contract,
                authorization=capability,
                no_registration_guard=guard,
                runtime_root=runtime,
            )
    assert not (evidence / "NO_ICP_ATTESTATION.json").exists()


def test_summary_has_30_evidence_derived_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    evidence.mkdir()
    runtime.mkdir()
    preprocessing = tmp_path / "preprocessing.json"
    backend = tmp_path / "backend.json"
    (runtime / "evidence").mkdir()
    tests = runtime / "evidence/full_test_status.json"
    preprocessing_value = {
        "crop_policy": {"kind": "NONE"},
        "deskew": {"uses_lidar_odometry": False},
        "dynamic_object_policy": {"kind": "NONE"},
        "execution_state_at_freeze": {"geometry_metric_execution_count": 0},
        "filtering": {"finite": True},
        "map_accumulation": {"voxel_m": 0.25},
        "source_downsampling": {"voxel_m": 0.1},
        "target_geometry_analysis": {"radius_m": 1.0},
    }
    atomic_write_json(preprocessing, preprocessing_value)
    atomic_write_json(backend, {"schema": "backend"})
    atomic_write_json(
        tests,
        {"collected": 10, "errors": 0, "failed": 0, "passed": 9, "schema": "zprm.boreas.v2.stage2.full_test_status.v1", "skipped": 1, "status": "PASS"},
    )
    atomic_write_json(
        evidence / "LIDAR_DOWNLOAD_AUDIT.json",
        {"raw_payload_persistent_bytes": 0, "successful_download_event_count": 20_161, "successful_payload_bytes": 123, "unique_allowlist_object_count": 20_061},
    )
    atomic_write_json(evidence / "MAP_LIDAR_DOWNLOAD_AUDIT.json", {"stream_deleted_raw_bytes": 23})
    atomic_write_json(evidence / "QUERY_LIDAR_DOWNLOAD_AUDIT.json", {"successful_payload_bytes": 100})
    atomic_write_json(evidence / "map_lineage_manifest.json", {"query_contribution_count": 0, "source_object_count": 8_202})
    atomic_write_json(evidence / "target_map_freeze_manifest.json", {"physical_target_map_copy_count": 1, "target_map_size_bytes": 456})
    atomic_write_json(evidence / "boreas_v2_stage2_selection_manifest.json", {"R14_frozen": True, "authority_bindings": {"primary_pair": {"map_sequence_id": "map", "query_sequence_id": "query"}}, "selection_manifest_sha256": "a" * 64})
    no_icp = {field: 0 for field in ("actual_open3d_trials", "actual_pcl_trials", "actual_trials", "open3d_registration_call_count", "other_registration_process_count", "pcl_cli_invocation_count", "real_trial_result_count", "registration_execution_count")}
    no_icp["pass"] = True
    atomic_write_json(evidence / "NO_ICP_ATTESTATION.json", no_icp)
    atomic_write_json(evidence / "boreas_v2_stage2_uncertainty_budget.json", {"rows": [{"component": name, "value": "UNKNOWN"} for name in closure.UNKNOWN_COMPONENTS]})
    atomic_write_csv(evidence / "all_candidate_scans.csv", [{"geometry_valid": "False"}] + [{"geometry_valid": "True"}] * 11_858, ("geometry_valid",))
    atomic_write_csv(evidence / "all_candidate_intervals.csv", [{"id": index} for index in range(246)], ("id",))
    atomic_write_csv(evidence / "selected_scene_intervals.csv", [{"scene_label": "CORRIDOR_OR_WEAK_GEOMETRY"}] * 10 + [{"scene_label": "GEOMETRY_RICH"}] * 10, ("scene_label",))
    atomic_write_csv(evidence / "selected_snapshots.csv", [{"scene_label": "CORRIDOR_OR_WEAK_GEOMETRY"}] * 50 + [{"scene_label": "GEOMETRY_RICH"}] * 50, ("scene_label",))
    canonical = [{"future_open3d_source_sha256": "1" * 64, "future_pcl_source_sha256": "1" * 64, "future_open3d_target_sha256": "2" * 64, "future_pcl_target_sha256": "2" * 64, "backend_parameter_contract_sha256": sha256_file(backend), "byte_identical_for_both_backends": "True"} for _ in range(100)]
    atomic_write_csv(evidence / "canonical_input_manifest.csv", canonical, tuple(canonical[0]))
    monkeypatch.setattr(closure, "_resource_evidence", lambda **_: {"verified": True})
    summary, markdown = closure._summary(
        evidence=evidence,
        runtime=runtime,
        preprocessing_contract_path=preprocessing,
        backend_contract_path=backend,
        reducer_resource_plan_path=runtime / "checkpoints/reducer_resource_plan.json",
        test_status_path=tests,
        disk_gate_events=({"operation": "START", "pass": True, "current_free_bytes": 999},),
    )
    assert [row["item"] for row in summary["answers"]] == list(range(1, 31))
    assert summary["empirical_counts_and_bytes"]["geometry_invalid_scan_count"] == 1
    assert summary["answers"][14]["answer"] == 11_859
    assert summary["answers"][26]["answer"]["test_status"]["passed"] == 9
    assert markdown.count(b"\n") > 60


def test_generated_document_resource_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "closure"
    runtime = tmp_path / "runtime"
    (runtime / "evidence").mkdir(parents=True)
    root.mkdir()
    authority = _authority(tmp_path)
    readiness = closure._readiness()
    atomic_write_json(root / "boreas_v2_stage2_readiness.json", readiness)
    rows = [{field: "UNKNOWN" for field in UNCERTAINTY_FIELDS} for _ in closure.UNKNOWN_COMPONENTS]
    for row, component in zip(rows, closure.UNKNOWN_COMPONENTS):
        row["component"] = component
    atomic_write_json(root / "boreas_v2_stage2_uncertainty_budget.json", {"rows": rows})
    no_icp = {field: 0 for field in ("actual_open3d_trials", "actual_pcl_trials", "actual_trials", "estimated_transform_count", "estimated_transform_file_count", "open3d_registration_call_count", "other_registration_process_count", "pcl_cli_invocation_count", "real_trial_result_count", "registration_execution_count", "structured_result_scan_error_count")}
    no_icp.update({"estimated_transform_evidence": [], "estimated_transform_files": [], "structured_result_scan_error_files": [], "pass": True, "status": "PASS"})
    atomic_write_json(root / "NO_ICP_ATTESTATION.json", no_icp)
    atomic_write_bytes(root / "boreas_v2_stage2_preprocessing_contract.md", authority.preprocessing_contract_path.with_suffix(".md").read_bytes())
    context = {
        "capacity_measurement": {"schema": "measurement"},
        "capacity_measurement_provenance": {"schema": "provenance"},
        "resource_plan": {"schema": "plan"},
    }
    for name, value in list(context.items()):
        filename = {"capacity_measurement": "target_context_capacity_measurement.json", "capacity_measurement_provenance": "target_context_capacity_measurement_provenance.json", "resource_plan": "target_context_resource_plan.json"}[name]
        path = runtime / "evidence" / filename
        atomic_write_json(path, value)
        context[f"{name}_file_sha256"] = sha256_file(path)
    summary = {
        "answers": [{"answer": True, "item": index, "question": "q"} for index in range(1, 31)],
        "conclusion": "x",
        "empirical_counts_and_bytes": {},
        "large_artifact_policy": "x",
        "readiness": readiness,
        "resource_capacity_evidence": {"reducer": {}, "runtime_reverification_dependencies": {}, "target_context": context, "target_map_content_address": {}},
        "schema": "zprm.boreas.v2.stage2.preparation_summary.v1",
    }
    atomic_write_json(root / "boreas_v2_stage2_summary.json", summary)
    atomic_write_json(runtime / "evidence/target_context_resource_plan.json", {"schema": "tampered"}, overwrite=True)
    with pytest.raises(closure.BoreasV2Stage2ClosureError, match="target-context evidence changed"):
        closure._verify_generated_documents(root, runtime=runtime, authority=authority)
