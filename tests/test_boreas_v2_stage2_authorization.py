from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from phase_a_harness.real_data_preparation import boreas_v2_stage2_authorization as auth
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard
from phase_a_harness.real_data_preparation.io import canonical_json_bytes
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier import (
    PAYLOAD_FILES,
    REQUIRED_FILES,
)


REPOSITORY = Path(__file__).resolve().parents[1]
HEAD_A = "a" * 40
HEAD_B = "b" * 40


def test_stage1_manifest_frozen_file_sha_constant_is_exact() -> None:
    path = (
        REPOSITORY
        / "frozen_assets/public_data_external_validation_v2_boreas_stage1/frozen_manifest.json"
    )
    assert len(auth.EXPECTED_STAGE1_MANIFEST_FILE_SHA256) == 64
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "71cfd78aa588c67dd28f8f8be87b7514c20ed090e75a8433583362253a97a8be"
    )
    assert auth.EXPECTED_STAGE1_MANIFEST_FILE_SHA256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _stage1_report() -> dict[str, Any]:
    return {
        "BOREAS_EXTERNAL_V2_STAGE1_READY": True,
        "BOREAS_EXTERNAL_V2_STAGE1_VERIFICATION_PASS": True,
        "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD": True,
        "manifest_root_sha256": auth.EXPECTED_STAGE1_MANIFEST_ROOT_SHA256,
        "primary_pair": {
            "map_sequence_id": auth.EXPECTED_MAP_SEQUENCE,
            "query_sequence_id": auth.EXPECTED_QUERY_SEQUENCE,
        },
        "registration_execution_count": 0,
        "selected_stage2_lidar_object_count": auth.EXPECTED_OBJECT_COUNT,
        "verification_pass": True,
    }


def _storage_report() -> dict[str, Any]:
    return {
        "BOREAS_V2_STAGE2_STORAGE_VERIFICATION_PASS": True,
        "CURRENT_DISK_SUFFICIENT": True,
        "STAGE2_STORAGE_PLAN_READY": True,
        "STAGE2_STORAGE_PLAN_VERIFICATION_PASS": True,
        "allowlist_object_count": auth.EXPECTED_OBJECT_COUNT,
        "allowlist_remote_bytes": auth.EXPECTED_REMOTE_BYTES,
        "lidar_payload_download_count": 0,
        "local_lidar_bin_count": 0,
        "manifest_root_sha256": auth.EXPECTED_STORAGE_MANIFEST_ROOT_SHA256,
        "real_execution_count": 0,
        "registration_execution_count": 0,
        "verification_pass": True,
    }


def _roots(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "stage1_data"
    runtime = tmp_path / "runtime"
    temporary = runtime / "tmp_download"
    for path in (data, runtime, temporary):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "data_root": data,
        "runtime_root": runtime,
        "temporary_root": temporary,
        "monitored_disk_path": runtime,
    }


def _patch_live_gates(
    monkeypatch: pytest.MonkeyPatch, *, head: dict[str, str] | None = None
) -> dict[str, str]:
    state = head or {"value": HEAD_A}

    def fake_git(_repository: Path, *arguments: str) -> str:
        if arguments == ("branch", "--show-current"):
            return auth.EXPECTED_BRANCH
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return state["value"]
        raise AssertionError(arguments)

    monkeypatch.setattr(auth, "_git", fake_git)
    monkeypatch.setattr(
        auth,
        "_authenticate_frozen_inputs",
        lambda _repository: {
            "stage1_manifest": auth.EXPECTED_STAGE1_MANIFEST_FILE_SHA256,
            "storage_manifest": auth.EXPECTED_STORAGE_MANIFEST_FILE_SHA256,
            "primary_pair": auth.EXPECTED_PAIR_SHA256,
            "allowlist": auth.EXPECTED_ALLOWLIST_SHA256,
            "storage_contract": auth.EXPECTED_STORAGE_CONTRACT_SHA256,
            "storage_budget": auth.EXPECTED_STORAGE_BUDGET_SHA256,
        },
    )
    monkeypatch.setattr(
        auth,
        "_authenticate_preprocessing",
        lambda _repository: (
            {"contract_payload_sha256": "d" * 64},
            "e" * 64,
        ),
    )
    monkeypatch.setattr(
        auth,
        "verify_boreas_external_v2_stage1",
        lambda **_kwargs: _stage1_report(),
    )
    monkeypatch.setattr(
        auth,
        "verify_boreas_v2_stage2_storage_plan",
        lambda **_kwargs: _storage_report(),
    )
    monkeypatch.setattr(
        auth,
        "_static_audit",
        lambda _repository: {"pass": True, "python_file_count": 42, "violations": []},
    )
    monkeypatch.setattr(
        auth.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=auth.MINIMUM_START_FREE_BYTES + 1),
    )
    return state


def _build(
    tmp_path: Path,
    guard: NoRegistrationGuard,
    *,
    stage1_report: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    roots = _roots(tmp_path)
    output = roots["runtime_root"] / "authorization.json"
    value = auth.build_boreas_v2_stage2_download_authorization(
        repository=REPOSITORY,
        **roots,
        stage1_verification_report=stage1_report,
        output_path=output,
        no_registration_guard=guard,
        now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
    )
    return value, roots, output


def _verify(
    roots: dict[str, Path], output: Path, guard: NoRegistrationGuard
) -> auth.VerifiedStage2Authorization:
    return auth.verify_boreas_v2_stage2_download_authorization(
        repository=REPOSITORY,
        **roots,
        authorization_path=output,
        no_registration_guard=guard,
    )


def _resign(path: Path, mutation: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    unsigned = {
        key: child
        for key, child in value.items()
        if key != "authorization_payload_sha256"
    }
    value["authorization_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    path.write_bytes(canonical_json_bytes(value))


def _publication_candidate(runtime: Path) -> Path:
    candidate = runtime / "checkpoints/final_closure_candidate"
    candidate.mkdir(parents=True)
    for name in sorted(PAYLOAD_FILES):
        (candidate / name).write_bytes(f"fixture:{name}\n".encode())
    manifest_core = {
        "payload": [
            {
                "path": name,
                "sha256": hashlib.sha256((candidate / name).read_bytes()).hexdigest(),
                "size_bytes": (candidate / name).stat().st_size,
            }
            for name in sorted(PAYLOAD_FILES)
        ]
    }
    manifest = {
        **manifest_core,
        "manifest_root_sha256": auth.compact_sha256(manifest_core),
    }
    (candidate / "boreas_v2_stage2_frozen_manifest.json").write_bytes(
        canonical_json_bytes(manifest)
    )
    (candidate / "SHA256SUMS").write_bytes(b"fixture sums\n")
    assert {path.name for path in candidate.iterdir()} == set(REQUIRED_FILES)
    return candidate


def test_publication_resume_git_exception_is_candidate_exact_and_single_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    candidate = _publication_candidate(runtime)
    destination = repository / "frozen_assets/boreas_v2_stage2_preparation"
    destination.parent.mkdir()
    import shutil

    shutil.copytree(candidate, destination)
    status = {
        "value": "\n".join(
            f"?? frozen_assets/boreas_v2_stage2_preparation/{name}"
            for name in sorted(REQUIRED_FILES)
        )
    }

    def fake_git(_repository: Path, *arguments: str) -> str:
        if arguments == ("branch", "--show-current"):
            return auth.EXPECTED_BRANCH
        if arguments == ("rev-parse", "HEAD"):
            return HEAD_A
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return status["value"]
        raise AssertionError(arguments)

    monkeypatch.setattr(auth, "_git", fake_git)
    state = auth._git_state_allowing_exact_closure_publication(repository, runtime)
    assert state == {"branch": auth.EXPECTED_BRANCH, "commit": HEAD_A, "status": ""}

    (destination / next(iter(REQUIRED_FILES))).write_bytes(b"tampered")
    with pytest.raises(auth.BoreasStage2AuthorizationError, match="bytes differ"):
        auth._git_state_allowing_exact_closure_publication(repository, runtime)
    shutil.rmtree(destination)
    shutil.copytree(candidate, destination)
    unrelated = repository / "unrelated.txt"
    unrelated.write_text("x")
    status["value"] += "\n?? unrelated.txt"
    with pytest.raises(auth.BoreasStage2AuthorizationError, match="unrelated"):
        auth._git_state_allowing_exact_closure_publication(repository, runtime)


def test_verify_capability_allows_only_exact_final_or_intent_bound_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    repository = tmp_path / "repository"
    repository.mkdir()
    roots = _roots(tmp_path)
    output = roots["runtime_root"] / "authorization.json"
    with NoRegistrationGuard() as guard:
        auth.build_boreas_v2_stage2_download_authorization(
            repository=repository,
            **roots,
            output_path=output,
            no_registration_guard=guard,
            now=lambda: datetime(2026, 8, 13, tzinfo=timezone.utc),
        )
        candidate = _publication_candidate(roots["runtime_root"])
        destination = repository / "frozen_assets/boreas_v2_stage2_preparation"
        destination.parent.mkdir()
        import shutil

        shutil.copytree(candidate, destination)
        status = {
            "value": "\n".join(
                f"?? frozen_assets/boreas_v2_stage2_preparation/{name}"
                for name in sorted(REQUIRED_FILES)
            )
        }

        def fake_git(_repository: Path, *arguments: str) -> str:
            if arguments == ("branch", "--show-current"):
                return auth.EXPECTED_BRANCH
            if arguments == ("rev-parse", "HEAD"):
                return HEAD_A
            if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                return status["value"]
            raise AssertionError(arguments)

        monkeypatch.setattr(auth, "_git", fake_git)
        capability = auth.verify_boreas_v2_stage2_download_authorization(
            repository=repository,
            **roots,
            authorization_path=output,
            no_registration_guard=guard,
            allow_exact_closure_publication_resume=True,
        )
        assert capability.formally_verified is True
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="clean worktree"):
            auth.verify_boreas_v2_stage2_download_authorization(
                repository=repository,
                **roots,
                authorization_path=output,
                no_registration_guard=guard,
            )

        shutil.rmtree(destination)
        manifest = json.loads(
            (candidate / "boreas_v2_stage2_frozen_manifest.json").read_text()
        )
        root_sha = manifest["manifest_root_sha256"]
        staging = destination.parent / f".{destination.name}.{root_sha}.staging"
        shutil.copytree(candidate, staging)
        intent_unsigned = {
            "candidate_root": str(candidate),
            "destination_root": str(destination),
            "manifest_root_sha256": root_sha,
            "payload": [
                {
                    "path": name,
                    "sha256": hashlib.sha256((candidate / name).read_bytes()).hexdigest(),
                    "size_bytes": (candidate / name).stat().st_size,
                }
                for name in sorted(REQUIRED_FILES)
            ],
            "runtime_root": str(roots["runtime_root"]),
            "schema": "zprm.boreas.v2.stage2.closure_publication_intent.v1",
            "staging_root": str(staging),
        }
        (roots["runtime_root"] / "checkpoints/closure_publication_intent.json").write_bytes(
            canonical_json_bytes(
                {
                    **intent_unsigned,
                    "intent_sha256": auth.compact_sha256(intent_unsigned),
                }
            )
        )
        status["value"] = "\n".join(
            f"?? frozen_assets/{staging.name}/{name}"
            for name in sorted(REQUIRED_FILES)
        )
        assert auth.verify_boreas_v2_stage2_download_authorization(
            repository=repository,
            **roots,
            authorization_path=output,
            no_registration_guard=guard,
            allow_exact_closure_publication_resume=True,
        ).formally_verified
        tampered_name = next(iter(sorted(REQUIRED_FILES)))
        (staging / tampered_name).write_bytes(b"tampered")
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="bytes differ"):
            auth.verify_boreas_v2_stage2_download_authorization(
                repository=repository,
                **roots,
                authorization_path=output,
                no_registration_guard=guard,
                allow_exact_closure_publication_resume=True,
            )
        (staging / tampered_name).write_bytes((candidate / tampered_name).read_bytes())

        shutil.copytree(candidate, destination)
        final_status = "\n".join(
            f"?? frozen_assets/boreas_v2_stage2_preparation/{name}"
            for name in sorted(REQUIRED_FILES)
        )
        staging_status = status["value"]
        status["value"] = f"{final_status}\n{staging_status}"
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="multiple roots"):
            auth.verify_boreas_v2_stage2_download_authorization(
                repository=repository,
                **roots,
                authorization_path=output,
                no_registration_guard=guard,
                allow_exact_closure_publication_resume=True,
            )
        shutil.rmtree(destination)
        status["value"] = staging_status
        status["value"] += "\n?? unrelated.txt"
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="unrelated"):
            auth.verify_boreas_v2_stage2_download_authorization(
                repository=repository,
                **roots,
                authorization_path=output,
                no_registration_guard=guard,
                allow_exact_closure_publication_resume=True,
            )


def test_build_recomputes_and_freezes_stage1_then_mints_live_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        value, roots, output = _build(tmp_path, guard)
        report_path = roots["runtime_root"] / "evidence" / (
            "boreas_v2_stage1_verification_live.json"
        )
        assert report_path.read_bytes() == canonical_json_bytes(_stage1_report())
        assert value["stage1_verification_report_path"] == str(report_path)
        assert value["stage1_verification_report_sha256"] == hashlib.sha256(
            canonical_json_bytes(_stage1_report())
        ).hexdigest()
        capability = _verify(roots, output, guard)
        assert isinstance(capability, auth.VerifiedStage2Authorization)
        assert capability.no_registration_guard is guard
        assert capability.document == value
        assert value["self_hash_semantics"] == auth.SELF_HASH_SEMANTICS
        assert value["REAL_REGISTRATION_AUTHORIZED"] is False
        assert value["PUBLIC_DATA_V2_RUN_AUTHORIZED"] is False
        with pytest.raises(TypeError, match="non-serializable"):
            pickle.dumps(capability)
        _resign(output, lambda value: value.__setitem__("actual_trials", False))
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="artifact changed"):
            capability.assert_operation_live()
    with pytest.raises(auth.BoreasStage2AuthorizationError, match="not active"):
        capability.assert_live()


def test_optional_external_stage1_report_is_only_an_exact_comparison_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    roots = _roots(tmp_path)
    forged = _stage1_report()
    forged["primary_pair"]["map_sequence_id"] = "forged"
    witness = roots["runtime_root"] / "forged-stage1.json"
    witness.write_bytes(canonical_json_bytes(forged))
    with NoRegistrationGuard() as guard:
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="supplied/independently"):
            _build(tmp_path, guard, stage1_report=witness)


def test_authorization_fails_closed_on_dirty_tree_and_low_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    clean_git = auth._git
    monkeypatch.setattr(
        auth,
        "_git",
        lambda repository, *arguments: (
            "untracked" if arguments[0] == "status" else clean_git(repository, *arguments)
        ),
    )
    with NoRegistrationGuard() as guard:
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="clean worktree"):
            _build(tmp_path, guard)

    _patch_live_gates(monkeypatch)
    monkeypatch.setattr(
        auth.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=auth.MINIMUM_START_FREE_BYTES - 1),
    )
    with NoRegistrationGuard() as guard:
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="BLOCKED_INSUFFICIENT_DISK"):
            _build(tmp_path, guard)


def test_exact_head_and_strict_types_cannot_be_bypassed_by_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        _, roots, output = _build(tmp_path, guard)
        state["value"] = HEAD_B
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="exact live HEAD"):
            _verify(roots, output, guard)
        state["value"] = HEAD_A
        _resign(output, lambda value: value.__setitem__("actual_trials", False))
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="authority projection"):
            _verify(roots, output, guard)


def test_resigned_live_authority_tamper_and_forged_capability_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        _, roots, output = _build(tmp_path, guard)
        _resign(
            output,
            lambda value: value.__setitem__(
                "preprocessing_contract_sha256", "f" * 64
            ),
        )
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="authority projection"):
            _verify(roots, output, guard)
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="only be minted"):
            auth.VerifiedStage2Authorization(
                _token=object(),
                _document_bytes=b"{}\n",
                repository=REPOSITORY,
                stage1_data_root=roots["data_root"],
                runtime_root=roots["runtime_root"],
                temporary_root=roots["temporary_root"],
                monitored_disk_path=roots["monitored_disk_path"],
                authorization_path=output,
                authorization_file_sha256="0" * 64,
                no_registration_guard=guard,
            )


def test_build_requires_the_exact_active_guard_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    guard = NoRegistrationGuard()
    roots = _roots(tmp_path)
    with pytest.raises(auth.BoreasStage2AuthorizationError, match="not active"):
        auth.build_boreas_v2_stage2_download_authorization(
            repository=REPOSITORY,
            **roots,
            output_path=roots["runtime_root"] / "authorization.json",
            no_registration_guard=guard,
        )


def test_root_symlink_is_rejected_before_any_live_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    roots = _roots(tmp_path)
    linked = tmp_path / "linked-runtime"
    linked.symlink_to(roots["runtime_root"], target_is_directory=True)
    with NoRegistrationGuard() as guard:
        with pytest.raises(auth.BoreasStage2AuthorizationError, match="non-symlink"):
            auth.build_boreas_v2_stage2_download_authorization(
                repository=REPOSITORY,
                data_root=roots["data_root"],
                runtime_root=linked,
                temporary_root=roots["temporary_root"],
                monitored_disk_path=roots["monitored_disk_path"],
                output_path=roots["runtime_root"] / "authorization.json",
                no_registration_guard=guard,
            )


def test_live_preprocessing_contract_accepts_only_the_frozen_witness_claim() -> None:
    contract, physical_sha256 = auth._authenticate_preprocessing(REPOSITORY)
    assert contract["implementation_bindings"][
        "independent_canonical_source_witness"
    ]["verification_claim"] == (
        "DUAL_PATH_BYTE_IDENTITY_USING_SHARED_FROZEN_PREPROCESSING_PRIMITIVES"
    )
    assert physical_sha256 == hashlib.sha256(
        (REPOSITORY / "protocols/boreas_v2_stage2_preprocessing_contract.json").read_bytes()
    ).hexdigest()


def test_live_capability_uses_runtime_watermark_after_fresh_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_live_gates(monkeypatch)
    monkeypatch.setenv("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard() as guard:
        _, roots, output = _build(tmp_path, guard)
        monkeypatch.setattr(
            auth.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(
                free=auth.RUNTIME_LOW_DISK_WATERMARK_BYTES + 1
            ),
        )
        assert isinstance(_verify(roots, output, guard), auth.VerifiedStage2Authorization)
        monkeypatch.setattr(
            auth.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(
                free=auth.RUNTIME_LOW_DISK_WATERMARK_BYTES - 1
            ),
        )
        with pytest.raises(
            auth.BoreasStage2AuthorizationError, match="runtime watermark"
        ):
            _verify(roots, output, guard)
