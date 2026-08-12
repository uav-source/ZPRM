#!/usr/bin/env python3
"""Independently verify and publish the small Boreas v2 Stage-2 closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (  # noqa: E402
    EXPECTED_STORAGE_BUDGET_SHA256,
    VerifiedStage2Authorization,
    verify_boreas_v2_stage2_download_authorization,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_closure import (  # noqa: E402
    BoreasV2Stage2ClosureError,
    assemble_small_closure_candidate,
    recover_closure_publication_staging,
    verify_and_publish_small_closure,
    write_selection_prerequisites,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preparation_verifier import (  # noqa: E402
    PreparationVerificationAuthority,
    REQUIRED_FILES,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_query_runner import (  # noqa: E402
    QueryRuntimeLease,
)
from phase_a_harness.real_data_preparation.guard import NoRegistrationGuard  # noqa: E402
from phase_a_harness.real_data_preparation.io import canonical_json_bytes  # noqa: E402
from phase_a_harness.real_data_preparation.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from phase_a_harness.real_data_preparation.stage2_disk_gate import (  # noqa: E402
    Stage2DiskGate,
)

try:  # Bind installed Open3D registration entrypoints for the guard lifetime.
    import open3d as _open3d  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - source-only test environments
    _open3d = None


FORMAL_PYTHON = Path(
    "/home/lj/.local/share/degen-lio-micromamba/envs/"
    "degen-lio-zprm-py311/bin/python3.11"
)
FULL_TEST_STATUS_SCHEMA = "zprm.boreas.v2.stage2.full_test_status.v1"
FULL_TEST_INTENT_SCHEMA = "zprm.boreas.v2.stage2.full_test_gate_intent.v1"
FULL_TEST_RESULT_SCHEMA = "zprm.boreas.v2.stage2.full_test_gate_result.v1"
CHILD_GUARD_SCHEMA = "zprm.boreas.v2.stage2.full_test_child_no_registration.v2"
CHILD_GUARD_PLUGIN = (
    "phase_a_harness.real_data_preparation.stage2_pytest_no_registration"
)
SAFE_REPLAY_MARKER_SCHEMA = (
    "zprm.boreas.v2.stage2.formal_full_pytest_marker.v1"
)
SAFE_REPLAY_MARKER_PURPOSE = (
    "BOREAS_V2_STAGE2_DATA_PREPARATION_FULL_TEST_NO_REGISTRATION"
)
SAFE_REPLAY_EVENT_SCHEMA = (
    "zprm.boreas.v2.stage2.safe_fixture_replay_event.v1"
)
SAFE_REPLAY_CATALOG_SHA256 = (
    "5402064dcfa87d489541685f57723bca5fd2b8128ec13762590af00637ba5fb9"
)
EMPTY_SAFE_REPLAY_EVENT_CHAIN_SHA256 = "0" * 64
OPEN3D_BACKEND = "open3d_point_to_plane"
PCL_BACKEND = "pcl_point_to_plane"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_REPLAY_BYTES = 41_998_817_280
EXPECTED_SKIPPED_TESTS = (
    (
        "tests.test_formal_semantics_equivalence::test_formal_snapshot_validator_scientific_body_is_baseline_exact",
        "source-only ZIP excludes the historical Git baseline object",
    ),
    (
        "tests.test_formal_semantics_equivalence::test_reader_policy_and_authentication_primitives_are_static_baseline_equivalent",
        "source-only ZIP excludes the historical Git baseline object",
    ),
    (
        "tests.test_full_synthetic_protocol_assets_runner::test_authorization_rejects_forged_gate_report",
        "source-only ZIP excludes the historical Phase B raw results",
    ),
    (
        "tests.test_full_synthetic_protocol_assets_runner::test_phase_a_ideal_import_is_read_only_and_complete",
        "source-only ZIP excludes the historical formal Phase A results",
    ),
    (
        "tests.test_full_synthetic_protocol_assets_runner::test_phase_b_overlap_retains_all_published_snapshot_and_trial_ids",
        "source-only ZIP excludes the historical Phase B tag object",
    ),
    (
        "tests.test_runtime_lifecycle_qualification::test_fixed_pcl_v3_input_inventory_matches_contract",
        "/tmp/synthetic_confirmatory_v2_pcl_v3_requalification: source-only package: external historical qualification bundle unavailable",
    ),
    (
        "tests.test_synthetic_confirmatory_v2::test_frozen_model_and_backend_implementations_are_unchanged",
        "source-only ZIP excludes the historical v2 Git object",
    ),
    (
        "tests.test_synthetic_confirmatory_v2::test_v1_and_v2_manifests_are_explicitly_version_selected",
        "source-only ZIP excludes the historical v2 Git object",
    ),
    (
        "tests.test_synthetic_confirmatory_v2::test_v1_failure_archive_is_preserved",
        "source-only ZIP excludes the historical v1 failure bundle",
    ),
    (
        "tests.test_synthetic_confirmatory_v2::test_v2_static_seed_provenance_audit_is_exact_and_constructor_free",
        "source-only ZIP excludes the historical v2 Git object",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--monitored-disk-path", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=REPOSITORY / "frozen_assets/boreas_v2_stage2_preparation",
    )
    return parser


def _canonical_directory(path: Path, label: str) -> Path:
    value = path.resolve(strict=True)
    if path != value or path.is_symlink() or not path.is_dir():
        raise BoreasV2Stage2ClosureError(f"{label} must be a canonical directory")
    return value


def _run_checked(
    argv: list[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=None if env is None else dict(env),
    )


def _git_value(repository: Path, *arguments: str) -> bytes:
    result = _run_checked(["git", *arguments], cwd=repository)
    if result.returncode != 0:
        raise BoreasV2Stage2ClosureError(
            f"Git evidence failed: {' '.join(arguments)}"
        )
    return result.stdout


def _source_worktree_status(repository: Path) -> bytes:
    """Ignore only closed Stage-2 publication directories, never source edits."""

    raw = _git_value(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if not raw:
        return b""
    allowed_roots: set[Path] = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("?? "):
            raise BoreasV2Stage2ClosureError(
                "formal full-test gate requires an unchanged tracked source tree"
            )
        relative = Path(line[3:])
        parts = relative.parts
        if len(parts) < 3 or parts[0] != "frozen_assets":
            raise BoreasV2Stage2ClosureError(
                "formal full-test gate found an unrelated untracked file"
            )
        directory = parts[1]
        if directory == "boreas_v2_stage2_preparation":
            root = repository / parts[0] / directory
        elif (
            directory.startswith(".boreas_v2_stage2_preparation.")
            and directory.endswith(".staging")
        ):
            root = repository / parts[0] / directory
        else:
            raise BoreasV2Stage2ClosureError(
                "formal full-test gate found an unrelated frozen-assets file"
            )
        allowed_roots.add(root)
    for root in allowed_roots:
        if (
            root.is_symlink()
            or not root.is_dir()
            or root.resolve(strict=True) != root
            or {entry.name for entry in root.iterdir()} != set(REQUIRED_FILES)
        ):
            raise BoreasV2Stage2ClosureError(
                "published closure worktree exception is not a closed inventory"
            )
        for entry in root.iterdir():
            metadata = os.lstat(entry)
            if (
                entry.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise BoreasV2Stage2ClosureError(
                    "published closure worktree exception contains an unsafe file"
                )
    return b""


def _junit_projection(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise BoreasV2Stage2ClosureError("full-test JUnit XML is invalid") from error
    cases = list(root.iter("testcase"))
    identities = [
        f"{case.get('classname', '')}::{case.get('name', '')}" for case in cases
    ]
    if any(identity.startswith("::") or identity.endswith("::") for identity in identities):
        raise BoreasV2Stage2ClosureError("full-test JUnit has an empty test identity")
    if len(identities) != len(set(identities)):
        raise BoreasV2Stage2ClosureError("full-test JUnit test identities are duplicated")
    skipped_rows = sorted(
        (
            identities[index],
            str(skipped.get("message", "")),
        )
        for index, case in enumerate(cases)
        if (skipped := case.find("skipped")) is not None
    )
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    counts = {
        "collected": len(cases),
        "errors": errors,
        "failed": failed,
        "passed": len(cases) - skipped - failed - errors,
        "skipped": skipped,
    }
    if (
        counts["collected"] < 934
        or counts["passed"] <= 0
        or tuple(skipped_rows) != EXPECTED_SKIPPED_TESTS
    ):
        raise BoreasV2Stage2ClosureError(
            "full-test JUnit skip identities/reasons differ from the frozen source-only set"
        )
    return {
        **counts,
        "skipped_tests": [
            {"node_id": node_id, "reason": reason}
            for node_id, reason in skipped_rows
        ],
        "test_case_ids_sha256": hashlib.sha256(
            canonical_json_bytes(identities)
        ).hexdigest(),
    }


def _canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or os.lstat(path).st_nlink != 1
    ):
        raise BoreasV2Stage2ClosureError(f"{label} is absent or unsafe")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoreasV2Stage2ClosureError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise BoreasV2Stage2ClosureError(f"{label} is noncanonical")
    return value


def _self_hashed(unsigned: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(unsigned)
    value[field] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return value


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _compact_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_json_bytes(value)).hexdigest()


def _formal_safe_replay_marker(
    *,
    repository: Path,
    python: Path,
    git_head: str,
    event_log: Path,
) -> dict[str, Any]:
    """Recompute the exact marker independently of an existing marker file."""

    replay_path = repository / (
        "src/phase_a_harness/real_data_preparation/stage2_safe_fixture_replay.py"
    )
    lifecycle_path = repository / "src/phase_a_harness/runtime_lifecycle_fixture.py"
    for path, label in (
        (replay_path, "safe fixture replay implementation"),
        (lifecycle_path, "runtime lifecycle fixture implementation"),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or os.lstat(path).st_nlink != 1
        ):
            raise BoreasV2Stage2ClosureError(f"formal {label} is unsafe")
    unsigned = {
        "event_log_path": str(event_log),
        "formal_full_pytest": True,
        "formal_python_executable": str(python),
        "git_head": git_head,
        "no_registration": True,
        "purpose": SAFE_REPLAY_MARKER_PURPOSE,
        "registration_execution_count": 0,
        "repository_root": str(repository),
        "runtime_lifecycle_fixture_sha256": sha256_file(lifecycle_path),
        "safe_fixture_replay_sha256": sha256_file(replay_path),
        "schema": SAFE_REPLAY_MARKER_SCHEMA,
    }
    return {**unsigned, "marker_payload_sha256": _compact_sha256(unsigned)}


def _validate_safe_replay_marker(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    value = _canonical_json(path, label="formal safe-fixture replay marker")
    if value != dict(expected):
        raise BoreasV2Stage2ClosureError("formal safe-fixture replay marker differs")
    return value


def _validate_safe_replay_event_log(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or os.lstat(path).st_nlink != 1
    ):
        raise BoreasV2Stage2ClosureError("formal safe-replay event log is unsafe")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise BoreasV2Stage2ClosureError("formal safe-replay event log is partial")
    fields = {
        "actual_registration_execution_count",
        "backend",
        "condition",
        "event_sha256",
        "event_version",
        "planned_trial_id",
        "previous_event_sha256",
        "process_id",
        "replay_catalog_sha256",
        "replay_result_sha256",
        "safe_simulated_materialization_count",
        "sequence",
        "snapshot_id",
    }
    previous = EMPTY_SAFE_REPLAY_EVENT_CHAIN_SHA256
    counts: Counter[str] = Counter()
    event_count = 0
    for event_count, raw in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise BoreasV2Stage2ClosureError(
                "formal safe-replay event log has invalid JSON"
            ) from error
        if type(event) is not dict or raw != _compact_json_bytes(event):
            raise BoreasV2Stage2ClosureError(
                "formal safe-replay event log is noncanonical"
            )
        unsigned = dict(event)
        claim = unsigned.pop("event_sha256", None)
        if (
            set(event) != fields
            or event.get("event_version") != SAFE_REPLAY_EVENT_SCHEMA
            or event.get("sequence") != event_count
            or event.get("previous_event_sha256") != previous
            or event.get("actual_registration_execution_count") != 0
            or event.get("safe_simulated_materialization_count") != 1
            or event.get("backend") not in (OPEN3D_BACKEND, PCL_BACKEND)
            or not isinstance(event.get("condition"), str)
            or not event.get("condition")
            or not isinstance(event.get("planned_trial_id"), str)
            or not event.get("planned_trial_id")
            or not isinstance(event.get("snapshot_id"), str)
            or not event.get("snapshot_id")
            or event.get("replay_catalog_sha256")
            != SAFE_REPLAY_CATALOG_SHA256
            or type(event.get("process_id")) is not int
            or event["process_id"] <= 0
            or not isinstance(event.get("replay_result_sha256"), str)
            or SHA256_RE.fullmatch(event["replay_result_sha256"]) is None
            or claim != _compact_sha256(unsigned)
        ):
            raise BoreasV2Stage2ClosureError(
                "formal safe-replay event chain differs"
            )
        previous = str(claim)
        counts[str(event["backend"])] += 1
    return {
        "safe_fixture_replay_event_chain_head_sha256": previous,
        "safe_fixture_replay_event_count": event_count,
        "safe_fixture_replay_event_log_sha256": hashlib.sha256(payload).hexdigest(),
        "safe_simulated_materialization_count": event_count,
        "safe_simulated_open3d_materialization_count": counts[OPEN3D_BACKEND],
        "safe_simulated_pcl_materialization_count": counts[PCL_BACKEND],
    }


def _unlink_regular(path: Path, *, label: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or os.lstat(path).st_nlink != 1
    ):
        raise BoreasV2Stage2ClosureError(f"{label} is unsafe to remove")
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_test_status(
    *,
    report: Path,
    junit: Path,
    result_path: Path,
    child_attestation: Path,
    event_log: Path,
    expected_core: Mapping[str, Any],
) -> dict[str, Any]:
    result = _canonical_json(result_path, label="full-test result receipt")
    result_unsigned = dict(result)
    result_claim = result_unsigned.pop("result_sha256", None)
    if (
        result.get("schema") != FULL_TEST_RESULT_SCHEMA
        or result_claim
        != hashlib.sha256(canonical_json_bytes(result_unsigned)).hexdigest()
        or any(result.get(key) != value for key, value in expected_core.items())
        or result.get("exit_code") != 0
        or result.get("junit_sha256") != sha256_file(junit)
        or result.get("child_no_registration_attestation_sha256")
        != sha256_file(child_attestation)
        or result.get("safe_fixture_replay_event_log_sha256")
        != sha256_file(event_log)
    ):
        raise BoreasV2Stage2ClosureError("full-test result receipt differs")
    projection = _junit_projection(junit)
    value = _canonical_json(report, label="full-test status")
    expected = {
        **projection,
        **expected_core,
        "exit_code": 0,
        "child_no_registration_attestation_relative_path": child_attestation.relative_to(
            report.parents[1]
        ).as_posix(),
        "child_no_registration_attestation_sha256": sha256_file(child_attestation),
        "junit_relative_path": junit.relative_to(report.parents[1]).as_posix(),
        "junit_sha256": sha256_file(junit),
        "result_receipt_relative_path": result_path.relative_to(
            report.parents[1]
        ).as_posix(),
        "result_receipt_sha256": sha256_file(result_path),
        "safe_fixture_replay_event_log_sha256": sha256_file(event_log),
        "schema": FULL_TEST_STATUS_SCHEMA,
        "status": "PASS",
        "stderr_sha256": result["stderr_sha256"],
        "stdout_sha256": result["stdout_sha256"],
    }
    if value != expected:
        raise BoreasV2Stage2ClosureError("full-test status projection differs")
    return value


def _validate_child_guard_attestation(
    path: Path,
    *,
    plugin_path: Path,
    guard_path: Path,
    marker_path: Path,
    event_log_path: Path,
) -> dict[str, Any]:
    value = _canonical_json(path, label="full-test child NO-ICP attestation")
    expected_fields = {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_evidence",
        "estimated_transform_file_count",
        "estimated_transform_files",
        "guard_implementation_path",
        "guard_implementation_sha256",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pass",
        "pcl_cli_invocation_count",
        "plugin_path",
        "plugin_sha256",
        "pytest_exit_status_at_attestation",
        "real_trial_result_count",
        "registration_execution_count",
        "safe_fixture_replay_catalog_sha256",
        "safe_fixture_replay_event_chain_head_sha256",
        "safe_fixture_replay_event_count",
        "safe_fixture_replay_event_log_path",
        "safe_fixture_replay_event_log_sha256",
        "safe_fixture_replay_event_schema",
        "safe_fixture_replay_marker_path",
        "safe_fixture_replay_marker_sha256",
        "safe_simulated_materialization_count",
        "safe_simulated_open3d_materialization_count",
        "safe_simulated_pcl_materialization_count",
        "schema",
        "status",
        "structured_result_scan_error_count",
        "structured_result_scan_error_files",
    }
    zero_fields = {
        "actual_open3d_trials",
        "actual_pcl_trials",
        "actual_trials",
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "pytest_exit_status_at_attestation",
        "real_trial_result_count",
        "registration_execution_count",
        "structured_result_scan_error_count",
    }
    replay = _validate_safe_replay_event_log(event_log_path)
    replay_count_fields = {
        "safe_fixture_replay_event_count",
        "safe_simulated_materialization_count",
        "safe_simulated_open3d_materialization_count",
        "safe_simulated_pcl_materialization_count",
    }
    if (
        set(value) != expected_fields
        or any(
            type(value.get(field)) is not int or value.get(field) != 0
            for field in zero_fields
        )
        or value.get("estimated_transform_evidence") != []
        or value.get("estimated_transform_files") != []
        or value.get("structured_result_scan_error_files") != []
        or value.get("pass") is not True
        or value.get("status") != "PASS"
        or value.get("schema") != CHILD_GUARD_SCHEMA
        or value.get("plugin_path") != str(plugin_path)
        or value.get("plugin_sha256") != sha256_file(plugin_path)
        or value.get("guard_implementation_path") != str(guard_path)
        or value.get("guard_implementation_sha256") != sha256_file(guard_path)
        or value.get("safe_fixture_replay_catalog_sha256")
        != SAFE_REPLAY_CATALOG_SHA256
        or value.get("safe_fixture_replay_event_schema")
        != SAFE_REPLAY_EVENT_SCHEMA
        or value.get("safe_fixture_replay_event_log_path") != str(event_log_path)
        or value.get("safe_fixture_replay_marker_path") != str(marker_path)
        or value.get("safe_fixture_replay_marker_sha256")
        != sha256_file(marker_path)
        or any(type(value.get(field)) is not int for field in replay_count_fields)
        or any(value.get(field) != expected for field, expected in replay.items())
    ):
        raise BoreasV2Stage2ClosureError(
            "full-test child NO-ICP attestation differs"
        )
    return value


def run_fixed_full_test_gate(
    repository: Path,
    runtime: Path,
    *,
    _fault_hook: Any | None = None,
) -> Path:
    """Run the one formal Py3.11 full-suite command and bind its raw JUnit."""

    python = FORMAL_PYTHON.resolve(strict=True)
    version = _run_checked([str(python), "--version"], cwd=repository)
    if version.returncode != 0 or not version.stdout.startswith(b"Python 3.11."):
        raise BoreasV2Stage2ClosureError("formal Python 3.11 interpreter differs")
    status = _source_worktree_status(repository)
    evidence = runtime / "evidence"
    junit = evidence / "full_pytest_junit.xml"
    partial = runtime / "checkpoints/full_pytest_junit.xml.partial"
    child_attestation = evidence / "full_pytest_no_registration_attestation.json"
    child_partial = (
        runtime / "checkpoints/full_pytest_no_registration_attestation.json.partial"
    )
    child_scan_root = runtime / "checkpoints/full_pytest_no_registration_scan_root"
    if child_scan_root.exists():
        child_scan_root = _canonical_directory(
            child_scan_root, "full-test child NO-ICP scan root"
        )
        if any(child_scan_root.iterdir()):
            raise BoreasV2Stage2ClosureError(
                "full-test child NO-ICP scan root is not empty"
            )
    else:
        child_scan_root.mkdir(mode=0o700)
    plugin_path = (
        repository
        / "src/phase_a_harness/real_data_preparation/stage2_pytest_no_registration.py"
    ).resolve(strict=True)
    guard_path = (
        repository / "src/phase_a_harness/real_data_preparation/guard.py"
    ).resolve(strict=True)
    report = evidence / "full_test_status.json"
    intent_path = runtime / "checkpoints/full_test_gate_intent.json"
    result_path = runtime / "checkpoints/full_test_gate_result.json"
    marker = evidence / "full_pytest_safe_fixture_replay_marker.json"
    event_log = evidence / "full_pytest_safe_fixture_replay_events.jsonl"
    git_head = _git_value(repository, "rev-parse", "HEAD").decode().strip()
    marker_value = _formal_safe_replay_marker(
        repository=repository,
        python=python,
        git_head=git_head,
        event_log=event_log,
    )
    marker_payload = canonical_json_bytes(marker_value)
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    controlled_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTEST_ADDOPTS": "",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(repository / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSTARTUP": "",
        "PYTHONWARNINGS": "",
        "MAMBA_ROOT_PREFIX": "/home/lj/.local/share/degen-lio-micromamba",
        "TZ": "UTC",
        "ZPRM_REAL_DATA_PREP_NO_REGISTRATION": "1",
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST": "1",
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER": str(marker),
        "ZPRM_STAGE2_FORMAL_FULL_PYTEST_MARKER_SHA256": marker_sha256,
        "ZPRM_STAGE2_PYTEST_GUARD_ATTESTATION": str(child_partial),
        "ZPRM_STAGE2_PYTEST_GUARD_SCAN_ROOT": str(child_scan_root),
    }
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        CHILD_GUARD_PLUGIN,
        "-c",
        "pyproject.toml",
        f"--junitxml={partial}",
        "tests",
    ]
    core = {
        "command": command,
        "controlled_environment": controlled_environment,
        "git_head": git_head,
        "git_tree": _git_value(repository, "rev-parse", "HEAD^{tree}").decode().strip(),
        "child_no_registration_guard_implementation_path": guard_path.relative_to(
            repository
        ).as_posix(),
        "child_no_registration_guard_implementation_sha256": sha256_file(guard_path),
        "child_no_registration_plugin_path": plugin_path.relative_to(repository).as_posix(),
        "child_no_registration_plugin_sha256": sha256_file(plugin_path),
        "safe_fixture_replay_event_log_relative_path": event_log.relative_to(
            runtime
        ).as_posix(),
        "safe_fixture_replay_marker_relative_path": marker.relative_to(
            runtime
        ).as_posix(),
        "safe_fixture_replay_marker_sha256": marker_sha256,
        "pyproject_sha256": sha256_file(repository / "pyproject.toml"),
        "python_executable": str(python),
        "python_executable_sha256": sha256_file(python),
        "python_version": version.stdout.decode().strip(),
        "source_worktree_status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_file_inventory_sha256": hashlib.sha256(
            _git_value(
                repository,
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                _git_value(repository, "rev-parse", "HEAD").decode().strip(),
            )
        ).hexdigest(),
    }
    intent_unsigned = {
        **core,
        "junit_final_relative_path": junit.relative_to(runtime).as_posix(),
        "junit_partial_relative_path": partial.relative_to(runtime).as_posix(),
        "child_attestation_final_relative_path": child_attestation.relative_to(
            runtime
        ).as_posix(),
        "child_attestation_partial_relative_path": child_partial.relative_to(
            runtime
        ).as_posix(),
        "result_relative_path": result_path.relative_to(runtime).as_posix(),
        "schema": FULL_TEST_INTENT_SCHEMA,
        "status_relative_path": report.relative_to(runtime).as_posix(),
    }
    intent = _self_hashed(intent_unsigned, "intent_sha256")

    if (
        report.exists()
        and junit.exists()
        and child_attestation.exists()
        and result_path.exists()
        and marker.exists()
        and event_log.exists()
    ):
        _validate_safe_replay_marker(marker, expected=marker_value)
        _validate_child_guard_attestation(
            child_attestation,
            plugin_path=plugin_path,
            guard_path=guard_path,
            marker_path=marker,
            event_log_path=event_log,
        )
        _validate_test_status(
            report=report,
            junit=junit,
            result_path=result_path,
            child_attestation=child_attestation,
            event_log=event_log,
            expected_core=core,
        )
        if intent_path.exists():
            if _canonical_json(intent_path, label="full-test intent") != intent:
                raise BoreasV2Stage2ClosureError("full-test intent differs")
            _unlink_regular(intent_path, label="completed full-test intent")
        _unlink_regular(partial, label="completed full-test partial JUnit")
        _unlink_regular(
            child_partial, label="completed full-test child attestation partial"
        )
        return report
    if any(
        path.exists() or path.is_symlink()
        for path in (
            report,
            junit,
            child_attestation,
            result_path,
            marker,
            event_log,
        )
    ):
        if not intent_path.is_file() or _canonical_json(
            intent_path, label="full-test intent"
        ) != intent:
            raise BoreasV2Stage2ClosureError(
                "partial full-test final evidence lacks the exact intent"
            )
    if intent_path.exists():
        if _canonical_json(intent_path, label="full-test intent") != intent:
            raise BoreasV2Stage2ClosureError("full-test intent differs")
    else:
        atomic_write_json(intent_path, intent, overwrite=False)
    if marker.exists() or marker.is_symlink():
        _validate_safe_replay_marker(marker, expected=marker_value)
    else:
        atomic_write_json(marker, marker_value, overwrite=False)
    if _fault_hook is not None:
        _fault_hook("AFTER_FULL_TEST_INTENT")

    if result_path.exists():
        result = _canonical_json(result_path, label="full-test result receipt")
        candidate_junit = junit if junit.exists() else partial
        if not candidate_junit.is_file() or result.get("junit_sha256") != sha256_file(
            candidate_junit
        ):
            raise BoreasV2Stage2ClosureError(
                "durable full-test result lacks its exact JUnit bytes"
            )
        candidate_attestation = (
            child_attestation if child_attestation.exists() else child_partial
        )
        if (
            not candidate_attestation.is_file()
            or result.get("child_no_registration_attestation_sha256")
            != sha256_file(candidate_attestation)
        ):
            raise BoreasV2Stage2ClosureError(
                "durable full-test result lacks its child NO-ICP attestation"
            )
        _validate_child_guard_attestation(
            candidate_attestation,
            plugin_path=plugin_path,
            guard_path=guard_path,
            marker_path=marker,
            event_log_path=event_log,
        )
    else:
        _unlink_regular(partial, label="interrupted full-test partial JUnit")
        _unlink_regular(
            child_partial, label="interrupted full-test child attestation"
        )
        _unlink_regular(
            event_log, label="interrupted full-test safe-replay event log"
        )
        execution_environment = {
            name: os.environ[name]
            for name in ("PATH", "TMPDIR")
            if name in os.environ
        }
        execution_environment.update(controlled_environment)
        completed = _run_checked(
            command, cwd=repository, env=execution_environment
        )
        projection = _junit_projection(partial)
        _validate_child_guard_attestation(
            child_partial,
            plugin_path=plugin_path,
            guard_path=guard_path,
            marker_path=marker,
            event_log_path=event_log,
        )
        if (
            completed.returncode != 0
            or projection["collected"] < 934
            or projection["passed"] <= 0
            or projection["failed"] != 0
            or projection["errors"] != 0
        ):
            raise BoreasV2Stage2ClosureError("formal full Py3.11 test suite failed")
        if _fault_hook is not None:
            _fault_hook("AFTER_FULL_TEST_CHILD")
        result_unsigned = {
            **core,
            "child_no_registration_attestation_sha256": sha256_file(child_partial),
            "exit_code": completed.returncode,
            "junit_sha256": sha256_file(partial),
            "safe_fixture_replay_event_log_sha256": sha256_file(event_log),
            "schema": FULL_TEST_RESULT_SCHEMA,
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        }
        atomic_write_json(
            result_path,
            _self_hashed(result_unsigned, "result_sha256"),
            overwrite=False,
        )
        if _fault_hook is not None:
            _fault_hook("AFTER_FULL_TEST_RESULT")

    if not junit.exists():
        os.rename(partial, junit)
        descriptor = os.open(
            junit.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _fault_hook is not None:
            _fault_hook("AFTER_FULL_TEST_JUNIT_RENAME")
    if not child_attestation.exists():
        os.rename(child_partial, child_attestation)
        descriptor = os.open(
            child_attestation.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _fault_hook is not None:
            _fault_hook("AFTER_FULL_TEST_ATTESTATION_RENAME")
    _validate_child_guard_attestation(
        child_attestation,
        plugin_path=plugin_path,
        guard_path=guard_path,
        marker_path=marker,
        event_log_path=event_log,
    )
    projection = _junit_projection(junit)
    result = _canonical_json(result_path, label="full-test result receipt")
    value = {
        **projection,
        **core,
        "exit_code": 0,
        "child_no_registration_attestation_relative_path": child_attestation.relative_to(
            runtime
        ).as_posix(),
        "child_no_registration_attestation_sha256": sha256_file(child_attestation),
        "junit_relative_path": junit.relative_to(runtime).as_posix(),
        "junit_sha256": sha256_file(junit),
        "result_receipt_relative_path": result_path.relative_to(runtime).as_posix(),
        "result_receipt_sha256": sha256_file(result_path),
        "safe_fixture_replay_event_log_sha256": sha256_file(event_log),
        "schema": FULL_TEST_STATUS_SCHEMA,
        "status": "PASS",
        "stderr_sha256": result["stderr_sha256"],
        "stdout_sha256": result["stdout_sha256"],
    }
    atomic_write_json(report, value, overwrite=False)
    if _fault_hook is not None:
        _fault_hook("AFTER_FULL_TEST_STATUS")
    _validate_test_status(
        report=report,
        junit=junit,
        result_path=result_path,
        child_attestation=child_attestation,
        event_log=event_log,
        expected_core=core,
    )
    _unlink_regular(intent_path, label="completed full-test intent")
    return report


def _authenticate_replay_for_resume(runtime: Path) -> None:
    replay = runtime / "map/transformed_xyz.f64le"
    ledger = runtime / "map/replay_ledger.jsonl"
    for path, label in ((replay, "replay"), (ledger, "replay ledger")):
        metadata = os.lstat(path)
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or path.resolve(strict=True) != path
        ):
            raise BoreasV2Stage2ClosureError(f"finalization {label} is unsafe")
    metadata = os.lstat(replay)
    if (
        metadata.st_size != EXPECTED_REPLAY_BYTES
        or metadata.st_blocks * 512 < EXPECTED_REPLAY_BYTES
        or ledger.stat().st_size <= 0
    ):
        raise BoreasV2Stage2ClosureError(
            "finalization requires the exact nonsparse authenticated replay"
        )


def main() -> int:
    arguments = _parser().parse_args()
    repository = _canonical_directory(arguments.repository.resolve(strict=True), "repository")
    data_root = _canonical_directory(arguments.data_root.resolve(strict=True), "data root")
    runtime_root = _canonical_directory(arguments.runtime_root.resolve(strict=True), "runtime root")
    temporary_root = _canonical_directory(
        arguments.temporary_root.resolve(strict=True), "temporary root"
    )
    monitored = _canonical_directory(
        arguments.monitored_disk_path.resolve(strict=True), "monitored disk"
    )
    frozen_root = arguments.frozen_root
    expected_frozen = repository / "frozen_assets/boreas_v2_stage2_preparation"
    if frozen_root != expected_frozen:
        raise BoreasV2Stage2ClosureError(
            "production frozen root must be the exact repository frozen-assets path"
        )
    frozen_parent = _canonical_directory(expected_frozen.parent, "frozen-assets parent")
    if frozen_root.parent != frozen_parent:
        raise BoreasV2Stage2ClosureError("frozen destination parent differs")

    contract_json = repository / "protocols/boreas_v2_stage2_preprocessing_contract.json"
    contract_markdown = repository / "protocols/boreas_v2_stage2_preprocessing_contract.md"
    backend_contract = repository / "frozen_assets/backend_parameter_contract.json"
    budget = repository / (
        "frozen_assets/boreas_v2_stage2_storage_optimization/"
        "boreas_v2_stage2_disk_budget_optimized.json"
    )
    candidate = runtime_root / "checkpoints/final_closure_candidate"

    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard(open3d_module=_open3d) as guard, QueryRuntimeLease(
        runtime_root
    ):
        recover_closure_publication_staging(
            candidate_root=candidate,
            destination_root=frozen_root,
            runtime_root=runtime_root,
            live_check=lambda: (
                guard.active
                or (_ for _ in ()).throw(
                    BoreasV2Stage2ClosureError(
                        "NO-ICP guard became inactive during publication recovery"
                    )
                )
            ),
        )
        authorization = verify_boreas_v2_stage2_download_authorization(
            repository=repository,
            data_root=data_root,
            runtime_root=runtime_root,
            temporary_root=temporary_root,
            monitored_disk_path=monitored,
            authorization_path=arguments.authorization.resolve(strict=True),
            no_registration_guard=guard,
            allow_exact_closure_publication_resume=True,
        )
        if (
            not isinstance(authorization, VerifiedStage2Authorization)
            or authorization.formally_verified is not True
        ):
            raise BoreasV2Stage2ClosureError(
                "finalization requires a formal production authorization capability"
            )
        disk_gate = Stage2DiskGate.from_frozen_budget(
            monitored,
            budget_path=budget,
            expected_budget_sha256=EXPECTED_STORAGE_BUDGET_SHA256,
            audit_log_path=runtime_root / "checkpoints/disk_gate_events.jsonl",
            storage_mode="RECOMMENDED_OPERATIONAL",
        )
        authorization.bind_disk_gate(disk_gate)
        _authenticate_replay_for_resume(runtime_root)
        disk_gate.assert_resume(operation_id="boreas-v2-stage2-closure-resume")
        disk_gate.before_checkpoint(
            64 * 1024**2, checkpoint_id="BOREAS_STAGE2_FULL_PYTEST_EVIDENCE"
        )
        test_status_path = run_fixed_full_test_gate(repository, runtime_root)
        disk_gate.before_checkpoint(
            1024**2, checkpoint_id="BOREAS_STAGE2_SELECTION_PREREQUISITES"
        )
        write_selection_prerequisites(
            evidence_root=runtime_root / "evidence",
            preprocessing_contract_path=contract_json,
            authorization=authorization,
            no_registration_guard=guard,
            runtime_root=runtime_root,
        )
        disk_gate.before_checkpoint(
            512 * 1024**2, checkpoint_id="BOREAS_STAGE2_SMALL_CLOSURE_CANDIDATE"
        )
        authority = PreparationVerificationAuthority.production(repository)
        assemble_small_closure_candidate(
            candidate_root=candidate,
            evidence_root=runtime_root / "evidence",
            authorization_path=authorization.authorization_path,
            preprocessing_contract_json=contract_json,
            preprocessing_contract_markdown=contract_markdown,
            backend_contract_path=backend_contract,
            reducer_resource_plan_path=(
                runtime_root / "checkpoints/reducer_resource_plan.json"
            ),
            test_status_path=test_status_path,
            disk_gate_events=disk_gate.events,
            runtime_root=runtime_root,
            authority=authority,
        )

        def live_check() -> None:
            authorization.assert_operation_live()
            attestation = guard.attestation(runtime_root)
            if attestation.get("pass") is not True:
                raise BoreasV2Stage2ClosureError(
                    "live NO-ICP attestation failed during final publication"
                )

        disk_gate.before_checkpoint(
            512 * 1024**2, checkpoint_id="BOREAS_STAGE2_SMALL_CLOSURE_PUBLICATION"
        )
        publication = verify_and_publish_small_closure(
            candidate_root=candidate,
            destination_root=frozen_root,
            runtime_root=runtime_root,
            authority=authority,
            live_check=live_check,
            production_mode=True,
        )
        result = {
            "BOREAS_EXTERNAL_V2_STAGE2_READY": publication.verification_report[
                "BOREAS_EXTERNAL_V2_STAGE2_READY"
            ],
            "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION": (
                publication.verification_report[
                    "READY_FOR_SEPARATE_STAGE3_REGISTRATION_AUTHORIZATION"
                ]
            ),
            "frozen_root": str(publication.destination),
            "independent_verification": publication.verification_report,
            "manifest_root_sha256": publication.manifest_root_sha256,
            "registration_execution_count": 0,
            "reused_existing_destination": publication.reused_existing_destination,
            "runtime_replay_retention": (
                "REQUIRED_FOR_FUTURE_EXACT_TARGET_REVERIFICATION_DO_NOT_DELETE"
            ),
        }
        print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
