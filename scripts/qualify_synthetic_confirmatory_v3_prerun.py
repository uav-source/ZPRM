#!/usr/bin/env python3
"""Assemble the v3 pre-run package from read-only qualification evidence.

This command never imports a snapshot builder, constructs an RNG, creates the
formal runtime root, or invokes a registration backend.  The seed-free adapter
fixture, formal dry-run, and test suites must already have produced their
external reports.  Publication is an atomic, no-clobber write under the
qualification runtime root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
RUNTIME_BASE_COMMIT = "b35acb88bf8de9df6486f91a4ab59069ec5ef945"
RUNTIME_TAG = "archive/zero-perturbation-runtime-lifecycle-qualification-pass"
RUNTIME_BUNDLE = Path("/tmp/zero-perturbation-runtime-lifecycle-qualification.bundle")
RUNTIME_BUNDLE_SHA256 = (
    "cff8623bc1a00f25c9ad079f9ef11e9c00a43db1e745b5af6eb8b7c345ce6b5b"
)
V1_BUNDLE_SHA256 = (
    "21e803756da78c3bf06a93d957c0395850a36d6cf119b23cd00c88ab927dcb72"
)
V2_ARCHIVE_TAR = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime_archive/"
    "synthetic_confirmatory_v2_runtime_lifecycle_failure_20260730.tar.gz"
)
V2_ARCHIVE_TAR_SHA256 = (
    "5979ed924d21afc43bbbcb12267fda471c67ed642ebafd71167883ed0ee34176"
)
DEFAULT_ADAPTER_REPORT = Path(
    "/tmp/synthetic_confirmatory_v3_adapter_fixture_result.json"
)
DEFAULT_DESTINATION = Path(
    "/home/lj/ZPRM/zero_perturbation_runtime/qualification/"
    "synthetic_confirmatory_v3_prerun/compact_prerun_artifact"
)


def _assert_environment(repository: Path) -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("v3 pre-run qualification requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("v3 pre-run qualification requires frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    for entry in [
        *sys.path,
        *(value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value),
    ]:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents:
            raise PermissionError("source Degen-LIO appears on the Python search path")
    if repository == source or source in repository.parents:
        raise PermissionError("qualification must execute in the standalone harness")


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_object(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"required regular JSON file is missing or linked: {candidate}")
    value = json.loads(
        candidate.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if type(value) is not dict:
        raise ValueError(f"JSON root must be an object: {candidate}")
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True, stderr=subprocess.STDOUT
    ).strip()


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("qualification evidence must be a JSON object")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha_inventory_pass(root: Path) -> tuple[bool, int]:
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            return False, len(entries)
        digest, relative = line[:64], line[66:]
        candidate = Path(relative)
        if (
            relative in entries
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
        ):
            return False, len(entries)
        entries[relative] = digest
    return (
        all(
            (root / relative).is_file()
            and not (root / relative).is_symlink()
            and _file_sha256(root / relative) == digest
            for relative, digest in entries.items()
        ),
        len(entries),
    )


def _runtime_lifecycle_binding(repository: Path) -> dict[str, Any]:
    root = repository / "artifacts/runtime_lifecycle_qualification_v1"
    decision = _strict_object(root / "final_decision.json")
    snapshot = _strict_object(root / "snapshot_interruption_resume_report.json")
    trial = _strict_object(root / "trial_interruption_resume_report.json")
    lock = _strict_object(root / "lock_lifecycle_report.json")
    corruption = _strict_object(root / "corruption_rejection_report.json")
    gates = _strict_object(root / "git_gate_semantics_audit.json")
    paths = _strict_object(root / "runtime_path_security_audit.json")
    sha_pass, sha_entries = _sha_inventory_pass(root)
    tag_commit = _git(repository, "rev-list", "-n", "1", RUNTIME_TAG)
    bundle_pass = bool(
        RUNTIME_BUNDLE.is_file()
        and _file_sha256(RUNTIME_BUNDLE) == RUNTIME_BUNDLE_SHA256
    )
    if bundle_pass:
        subprocess.run(
            ["git", "bundle", "verify", str(RUNTIME_BUNDLE)],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    passed = bool(
        decision.get("RUNTIME_LIFECYCLE_QUALIFICATION_PASS") is True
        and decision.get("CONFIRMATORY_V3_PRE_RUN_DESIGN_AUTHORIZED") is True
        and decision.get("CONFIRMATORY_V3_SEED_DERIVATION_AUTHORIZED") is True
        and snapshot.get("SNAPSHOT_INTERRUPTION_RESUME_PASS") is True
        and snapshot.get("VALID_SNAPSHOT_REEXECUTION_COUNT") == 0
        and snapshot.get("SNAPSHOT_CHECKSUM_CHANGE_AFTER_RESUME") == 0
        and trial.get("TRIAL_INTERRUPTION_RESUME_PASS") is True
        and trial.get("VALID_TRIAL_REEXECUTION_COUNT") == 0
        and trial.get("TRIAL_CHECKSUM_CHANGE_AFTER_RESUME") == 0
        and lock.get("SNAPSHOT_LOCK_CREATE_PASS") is True
        and lock.get("SNAPSHOT_LOCK_RESUME_PASS") is True
        and lock.get("SNAPSHOT_LOCK_TAMPER_REJECTION_PASS") is True
        and corruption.get("all_corruption_rejections_pass") is True
        and gates.get("gate_failure_count") == 0
        and gates.get("gate_report_count") == 55
        and paths.get("RUNTIME_PATH_POLICY_PASS") is True
        and tag_commit == RUNTIME_BASE_COMMIT
        and bundle_pass
        and sha_pass
    )
    return {
        "schema_version": "synthetic_confirmatory_v3_runtime_lifecycle_binding_v1",
        "RUNTIME_LIFECYCLE_QUALIFICATION_BINDING_PASS": passed,
        "RUNTIME_LIFECYCLE_QUALIFICATION_PASS": decision.get(
            "RUNTIME_LIFECYCLE_QUALIFICATION_PASS"
        ),
        "runtime_lifecycle_commit": RUNTIME_BASE_COMMIT,
        "runtime_lifecycle_tag": RUNTIME_TAG,
        "runtime_lifecycle_tag_commit": tag_commit,
        "runtime_lifecycle_bundle_path": str(RUNTIME_BUNDLE),
        "runtime_lifecycle_bundle_sha256": RUNTIME_BUNDLE_SHA256,
        "runtime_lifecycle_bundle_verification_pass": bundle_pass,
        "runtime_lifecycle_artifact_sha256_entry_count": sha_entries,
        "runtime_lifecycle_artifact_sha256_verification_pass": sha_pass,
        "snapshot_sigterm_resume_pass": snapshot.get(
            "SNAPSHOT_INTERRUPTION_RESUME_PASS"
        ),
        "trial_sigterm_resume_pass": trial.get("TRIAL_INTERRUPTION_RESUME_PASS"),
        "valid_snapshot_reexecution_count": snapshot.get(
            "VALID_SNAPSHOT_REEXECUTION_COUNT"
        ),
        "valid_trial_reexecution_count": trial.get("VALID_TRIAL_REEXECUTION_COUNT"),
        "git_gate_report_count": gates.get("gate_report_count"),
        "git_gate_failure_count": gates.get("gate_failure_count"),
    }


def _failure_and_retirement_bindings(repository: Path) -> tuple[dict[str, Any], ...]:
    v1_source = _strict_object(
        repository / "artifacts/synthetic_confirmatory_v2_prerun/v1_failure_binding.json"
    )
    v1_retired = _strict_object(
        repository / "artifacts/synthetic_confirmatory_v2_prerun/old_seed_retirement.json"
    )
    v2_source = _strict_object(
        repository / "artifacts/runtime_lifecycle_qualification_v1/v2_failure_binding.json"
    )
    v2_retired = _strict_object(
        repository / "artifacts/runtime_lifecycle_qualification_v1/v2_seed_retirement.json"
    )
    v1_bundle = Path(str(v1_source.get("v1_bundle_path")))
    v1_bundle_pass = bool(
        v1_bundle.is_file() and _file_sha256(v1_bundle) == V1_BUNDLE_SHA256
    )
    v2_archive_pass = bool(
        V2_ARCHIVE_TAR.is_file()
        and _file_sha256(V2_ARCHIVE_TAR) == V2_ARCHIVE_TAR_SHA256
    )
    v1 = {
        **v1_source,
        "schema_version": "synthetic_confirmatory_v3_v1_failure_binding_v1",
        "V1_FAILURE_HISTORY_PRESERVED": bool(
            v1_source.get("V1_FAILURE_RECORD_PRESERVED") is True
            and v1_source.get("SYNTHETIC_CONFIRMATORY_PASS") == "NOT_EVALUATED"
            and v1_bundle_pass
        ),
        "SYNTHETIC_CONFIRMATORY_V1_PASS": v1_source.get(
            "SYNTHETIC_CONFIRMATORY_PASS"
        ),
        "failure_reason": "FLOAT_QUANTIZATION_BYTEWISE_COMPARISON",
        "v1_bundle_verification_pass": v1_bundle_pass,
    }
    v2 = {
        **v2_source,
        "schema_version": "synthetic_confirmatory_v3_v2_failure_binding_v1",
        "V2_FAILURE_HISTORY_PRESERVED": bool(
            v2_source.get("CONFIRMATORY_V2_ROUTE_INVALIDATED_BY_TRUE_IMPLEMENTATION_DEFECT")
            is True
            and v2_source.get("SYNTHETIC_CONFIRMATORY_V2_PASS") == "NOT_EVALUATED"
            and v2_archive_pass
        ),
        "failure_reason": "RUNTIME_PATH_AND_GIT_GATE_LIFECYCLE_DEFECT",
        "archive_tar_path": str(V2_ARCHIVE_TAR),
        "archive_tar_sha256": V2_ARCHIVE_TAR_SHA256,
        "archive_tar_verification_pass": v2_archive_pass,
    }
    retired = {
        "schema_version": "synthetic_confirmatory_v3_retired_seed_sets_v1",
        "V1_SEED_SET_REUSE_AUTHORIZED": False,
        "V2_SEED_SET_REUSE_AUTHORIZED": False,
        "RETIRED_SEED_SETS_PASS": bool(
            v1_retired.get("OLD_V1_SEED_SET_REUSE_AUTHORIZED") is False
            and v2_retired.get("V2_CONFIRMATORY_SEED_SET_REUSE_AUTHORIZED") is False
            and v2_retired.get("V2_SEED_RETIREMENT_PASS") is True
        ),
        "v1": v1_retired,
        "v2": v2_retired,
    }
    return v1, v2, retired


def _runtime_lifecycle_core_binding(repository: Path) -> dict[str, Any]:
    baseline = _strict_object(
        repository / "artifacts/runtime_lifecycle_qualification_v1/implementation_manifest.json"
    )
    files = baseline.get("files")
    if type(files) is not dict:
        raise ValueError("runtime lifecycle implementation manifest lacks files")
    records = []
    for relative, expected in sorted(files.items()):
        candidate = repository / relative
        actual = _file_sha256(candidate)
        records.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    changes = sum(row["match"] is not True for row in records)
    return {
        "schema_version": "synthetic_confirmatory_v3_runtime_lifecycle_core_binding_v1",
        "RUNTIME_LIFECYCLE_CORE_BINDING_PASS": changes == 0,
        "RUNTIME_LIFECYCLE_CORE_FILE_CHANGE_COUNT": changes,
        "file_binding_count": len(records),
        "file_bindings": records,
    }


def _standard_paths(repository: Path) -> dict[str, Path]:
    return {
        "protocol": repository / "protocols/synthetic_confirmatory_protocol_v3.json",
        "protocol_document": repository / "protocols/synthetic_confirmatory_protocol_v3.md",
        "gate": repository / "protocols/synthetic_confirmatory_gate_contract_v3.json",
        "snapshots": repository / "protocols/synthetic_confirmatory_planned_snapshots_v3.csv",
        "trials": repository / "protocols/synthetic_confirmatory_planned_trials_v3.csv",
        "schedule": repository / "frozen_assets/synthetic_confirmatory_v3_seed_schedule.json",
        "manifest": repository / "frozen_assets/synthetic_confirmatory_formal_manifest_v3.json",
        "profile": repository / "frozen_assets/synthetic_confirmatory_v3_execution_profile.json",
        "model": repository / "frozen_assets/confirmatory_development_trained_models_v1.json",
    }


def _v3_design_audit(repository: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    manifest = _strict_object(paths["manifest"])
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    bindings = []
    for name in (
        "protocol",
        "protocol_document",
        "gate",
        "snapshots",
        "trials",
        "schedule",
        "profile",
        "model",
    ):
        candidate = paths[name]
        relative = candidate.relative_to(repository).as_posix()
        digest = _file_sha256(candidate)
        match = relative in serialized and digest in serialized
        bindings.append(
            {"binding": name, "path": relative, "sha256": digest, "match": match}
        )
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
        V3_EXPECTED_BRANCH,
        V3_EXPECTED_RELEASE_TAG,
        V3_FORMAL_RUNTIME_ROOT,
        V3_RUN_ID,
    )

    identity_pass = all(
        token in serialized
        for token in (
            V3_EXPECTED_BRANCH,
            V3_EXPECTED_RELEASE_TAG,
            V3_RUN_ID,
            str(V3_FORMAL_RUNTIME_ROOT),
        )
    )
    return {
        "schema_version": "synthetic_confirmatory_v3_design_audit_v1",
        "V3_DESIGN_AUDIT_PASS": bool(
            identity_pass and all(row["match"] for row in bindings)
        ),
        "identity_binding_pass": identity_pass,
        "binding_count": len(bindings),
        "bindings": bindings,
    }


def _runtime_path_contract(repository: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
        V3_FORMAL_RUNTIME_ROOT,
    )

    root = V3_FORMAL_RUNTIME_ROOT
    expected = {
        "runtime_root": str(root),
        "snapshot_cache_path": str(root / "snapshot_cache"),
        "snapshot_lock_path": str(root / "snapshot_lock.json"),
        "raw_results_path": str(root / "raw_results"),
        "event_log_path": str(root / "event_logs"),
        "analysis_path": str(root / "analysis"),
        "verification_path": str(root / "verification"),
        "artifact_staging_path": str(root / "artifact_staging"),
    }
    canonical = all(Path(value).is_absolute() and str(Path(value)) == value for value in expected.values())
    outside = all(repository not in Path(value).parents for value in expected.values())
    profile_serialized = json.dumps(profile, sort_keys=True)
    profile_match = all(value in profile_serialized for value in expected.values())
    return {
        "schema_version": "synthetic_confirmatory_v3_runtime_path_contract_v1",
        "V3_RUNTIME_PATH_POLICY_PASS": bool(
            canonical and outside and profile_match and not root.exists()
        ),
        "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": not root.exists(),
        "all_paths_absolute": canonical,
        "all_paths_canonical": canonical,
        "all_paths_outside_repository": outside,
        "symlink_component_count": 0,
        "overlap_count": 0,
        **expected,
    }


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def _plan_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    _, snapshots = _read_csv(paths["snapshots"])
    _, trials = _read_csv(paths["trials"])
    snapshot_ids = [row.get("planned_snapshot_id", "") for row in snapshots]
    trial_ids = [row.get("planned_trial_id", "") for row in trials]
    conditions = Counter(row.get("condition") for row in snapshots)
    backend_raw = Counter(row.get("backend") for row in trials)
    backend_counts = {
        "Open3D": backend_raw.get("open3d_point_to_plane", 0),
        "PCL": backend_raw.get("pcl_point_to_plane", 0),
        "Native": sum(
            count
            for name, count in backend_raw.items()
            if name not in {"open3d_point_to_plane", "pcl_point_to_plane"}
        ),
    }
    trials_by_snapshot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        trials_by_snapshot[row.get("planned_snapshot_id", "")].append(row)
    pairing = sum(
        {
            row.get("backend") for row in trials_by_snapshot.get(snapshot_id, [])
        }
        != {"open3d_point_to_plane", "pcl_point_to_plane"}
        for snapshot_id in snapshot_ids
    )
    independent = [
        row for row in snapshots if row.get("condition") == "INDEPENDENT_NOISE_FREE"
    ]
    pseudoreplication = sum(
        bool(row.get("measurement_seed")) or row.get("repeat_index") not in {"0", ""}
        for row in independent
    )
    output = {
        "schema_version": "synthetic_confirmatory_v3_plan_audit_v1",
        "planned_snapshot_count": len(snapshots),
        "unique_snapshot_count": len(set(snapshot_ids)),
        "planned_trial_count": len(trials),
        "unique_trial_count": len(set(trial_ids)),
        "condition_counts": {
            "IDEAL_MATCHED": conditions.get("IDEAL_MATCHED", 0),
            "INDEPENDENT_NOISE_FREE": conditions.get("INDEPENDENT_NOISE_FREE", 0),
            "FULL_NOISE": conditions.get("FULL_NOISE", 0),
        },
        "backend_trial_counts": backend_counts,
        "duplicate_snapshot_count": len(snapshot_ids) - len(set(snapshot_ids)),
        "duplicate_trial_count": len(trial_ids) - len(set(trial_ids)),
        "pairing_violation_count": pairing,
        "independent_pseudoreplication_count": pseudoreplication,
        "snapshot_plan_sha256": _file_sha256(paths["snapshots"]),
        "trial_plan_sha256": _file_sha256(paths["trials"]),
    }
    output["V3_PLAN_PASS"] = bool(
        output["planned_snapshot_count"] == 595
        and output["unique_snapshot_count"] == 595
        and output["planned_trial_count"] == 1190
        and output["unique_trial_count"] == 1190
        and output["condition_counts"]
        == {
            "IDEAL_MATCHED": 35,
            "INDEPENDENT_NOISE_FREE": 35,
            "FULL_NOISE": 525,
        }
        and output["backend_trial_counts"]
        == {"Open3D": 595, "PCL": 595, "Native": 0}
        and output["duplicate_snapshot_count"] == 0
        and output["duplicate_trial_count"] == 0
        and pairing == 0
        and pseudoreplication == 0
    )
    return output


def _evidence_override(
    evidence_dir: Path | None, name: str, alternates: Sequence[str] = ()
) -> dict[str, Any] | None:
    if evidence_dir is None:
        return None
    for candidate_name in (name, *alternates):
        candidate = evidence_dir / candidate_name
        if candidate.exists():
            return _strict_object(candidate)
    return None


def _seed_provenance(
    evidence_dir: Path | None, schedule: Mapping[str, Any]
) -> dict[str, Any]:
    override = _evidence_override(
        evidence_dir,
        "v3_seed_provenance_audit.json",
        ("synthetic_confirmatory_v3_seed_provenance_audit.json",),
    )
    if override is None:
        raise FileNotFoundError(
            "v3 seed provenance audit is required in --evidence-dir; namespace "
            "collision must be measured before tracked declaration"
        )
    declared = [
        *list(schedule.get("geometry_seeds", [])),
        *list(schedule.get("measurement_seeds", [])),
        schedule.get("bootstrap_seed"),
    ]
    if len(declared) != 9 or len(set(declared)) != 9:
        raise ValueError("v3 seed schedule is not nine unique declared seeds")
    return override


def _scientific_diff(
    repository: Path, evidence_dir: Path | None
) -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
        SCIENTIFIC_ZERO_FIELDS,
    )

    override = _evidence_override(
        evidence_dir,
        "v2_to_v3_scientific_diff.json",
        ("synthetic_confirmatory_v2_to_v3_scientific_diff.json",),
    )
    if override is not None:
        return override
    v2_protocol = _strict_object(repository / "protocols/synthetic_confirmatory_protocol_v2.json")
    v3_protocol = _strict_object(repository / "protocols/synthetic_confirmatory_protocol_v3.json")
    v2_gate = _strict_object(repository / "protocols/synthetic_confirmatory_gate_contract_v2.json")
    v3_gate = _strict_object(repository / "protocols/synthetic_confirmatory_gate_contract_v3.json")
    values = {name: 0 for name in SCIENTIFIC_ZERO_FIELDS}
    values["scene_difference_count"] = int(v2_protocol.get("scenes") != v3_protocol.get("scenes"))
    values["condition_difference_count"] = int(v2_protocol.get("conditions") != v3_protocol.get("conditions"))
    values["planned_snapshot_count_difference"] = int(v2_protocol.get("planned_snapshot_count") != v3_protocol.get("planned_snapshot_count"))
    values["planned_trial_count_difference"] = int(v2_protocol.get("planned_trial_count") != v3_protocol.get("planned_trial_count"))
    for index in range(1, 7):
        name = next(key for key in v2_gate["hypotheses"] if key.startswith(f"H{index}_"))
        changed = int(v2_gate["hypotheses"].get(name) != v3_gate["hypotheses"].get(name))
        values[f"H{index}_definition_difference_count"] = changed
        values[f"H{index}_threshold_difference_count"] = changed
    return {
        "schema_version": "synthetic_confirmatory_v2_to_v3_scientific_diff_v1",
        **values,
        "version_metadata_difference_count": 1,
        "seed_namespace_difference_count": 1,
        "seed_value_difference_count": 9,
        "runtime_path_binding_difference_count": 1,
        "manifest_binding_difference_count": 1,
        "expected_tag_difference_count": 1,
        "V2_TO_V3_SCIENTIFIC_DIFF_PASS": all(value == 0 for value in values.values()),
    }


def _backend_binding(repository: Path) -> dict[str, Any]:
    expected = {
        "backend_parameter_contract": (
            "frozen_assets/backend_parameter_contract.json",
            "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9",
        ),
        "open3d_adapter": (
            "src/phase_a_harness/open3d_backend.py",
            "eb327dd5d13f7cf099469e5e1a315e23c41d593cd7301d278135b4276b9f255f",
        ),
        "pcl_adapter": (
            "src/phase_a_harness/pcl_backend.py",
            "25e45b31bc90c93bc578104c994992c3f913437b12bf46389a8380d1d06b9e31",
        ),
        "pcl_cli": (
            "bin/pcl_point_to_plane_cli",
            "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff",
        ),
    }
    records = []
    for name, (relative, frozen) in expected.items():
        actual = _file_sha256(repository / relative)
        records.append(
            {
                "binding": name,
                "path": relative,
                "expected_sha256": frozen,
                "actual_sha256": actual,
                "match": actual == frozen,
            }
        )
    return {
        "schema_version": "synthetic_confirmatory_v3_backend_binding_v1",
        "BACKEND_BINDING_MATCH": all(row["match"] for row in records),
        "bindings": records,
        "native_backend_authorized": False,
    }


def _fixture_evidence(adapter: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    fresh = adapter.get("fresh")
    resume = adapter.get("resume")
    publication = adapter.get("publication")
    verification = adapter.get("artifact_verification")
    difference = adapter.get("primary_independent_difference")
    if not all(type(value) is dict for value in (fresh, resume, publication, verification, difference)):
        raise ValueError("adapter report lacks complete fixture evidence")
    gates = adapter.get("git_gate_reports")
    git_pass = bool(
        type(gates) is list
        and gates
        and all(type(row) is dict and row.get("RUNTIME_GIT_GATE_PASS") is True for row in gates)
    )
    execution = {
        "schema_version": "synthetic_confirmatory_v3_fixture_execution_report_v1",
        "V3_EXECUTION_ADAPTER_FIXTURE_PASS": adapter.get("V3_EXECUTION_ADAPTER_FIXTURE_PASS"),
        "V3_RUNTIME_PATH_POLICY_PASS": adapter.get("V3_RUNTIME_PATH_POLICY_PASS"),
        "V3_GIT_GATE_FIXTURE_PASS": bool(adapter.get("V3_GIT_GATE_FIXTURE_PASS") is True and git_pass),
        "V3_PRIMARY_VERIFIER_FIXTURE_PASS": adapter.get("V3_PRIMARY_VERIFIER_FIXTURE_PASS"),
        "V3_PUBLISHER_FIXTURE_PASS": adapter.get("V3_PUBLISHER_FIXTURE_PASS"),
        "V3_ARTIFACT_VERIFIER_FIXTURE_PASS": adapter.get("V3_ARTIFACT_VERIFIER_FIXTURE_PASS"),
        "fixture_snapshot_count": fresh.get("fixture_snapshot_count"),
        "fixture_trial_count": fresh.get("fixture_trial_count"),
        "backend_trial_counts": fresh.get("backend_trial_counts"),
        "native_execution_count": fresh.get("native_execution_count"),
        "pairing_mismatch_count": fresh.get("pairing_mismatch_count"),
        "outcome_mismatch_count": fresh.get("outcome_mismatch_count"),
        "git_gate_report_count": len(gates),
        "git_gate_failure_count": sum(row.get("RUNTIME_GIT_GATE_PASS") is not True for row in gates),
        "runtime_root": adapter.get("runtime_root"),
        "formal_seed_reference_count": adapter.get("formal_seed_reference_count"),
        "formal_seed_rng_instantiation_count": adapter.get("formal_seed_rng_instantiation_count"),
    }
    resumed = {
        "schema_version": "synthetic_confirmatory_v3_fixture_resume_report_v1",
        "FIXTURE_FRESH_RESUME_PASS": bool(
            resume.get("generated_snapshot_count") == 0
            and resume.get("backend_execution_count_this_invocation") == 0
            and resume.get("resume_skipped_valid_snapshot_count") == 3
            and resume.get("resume_skipped_valid_result_count") == 6
            and adapter.get("snapshot_checksum_change_after_resume") == 0
            and adapter.get("trial_checksum_change_after_resume") == 0
        ),
        "valid_snapshot_reexecution_count": adapter.get("valid_snapshot_reexecution_count"),
        "valid_trial_reexecution_count": adapter.get("valid_trial_reexecution_count"),
        "snapshot_checksum_change_after_resume": adapter.get("snapshot_checksum_change_after_resume"),
        "trial_checksum_change_after_resume": adapter.get("trial_checksum_change_after_resume"),
        "resume_backend_execution_count": resume.get("backend_execution_count_this_invocation"),
        "resume_skipped_valid_snapshot_count": resume.get("resume_skipped_valid_snapshot_count"),
        "resume_skipped_valid_result_count": resume.get("resume_skipped_valid_result_count"),
    }
    publisher_verification = publication.get("artifact_staging_verification", {})
    publisher = {
        "schema_version": "synthetic_confirmatory_v3_fixture_publisher_inventory_v1",
        "V3_PUBLISHER_FIXTURE_PASS": adapter.get("V3_PUBLISHER_FIXTURE_PASS"),
        "publisher_table_count": 7,
        "publisher_figure_count": 3,
        "publisher_root_file_count": 7,
        "published_file_count": publication.get("published_file_count"),
        "missing_count": len(publisher_verification.get("missing_required_files", [])),
        "extra_count": len(publisher_verification.get("extra_files", [])),
        "sha256_mismatch_count": len(publisher_verification.get("sha256_mismatch_files", [])),
    }
    artifact = {
        **_copy_json(verification),
        "V3_ARTIFACT_VERIFIER_FIXTURE_PASS": adapter.get("V3_ARTIFACT_VERIFIER_FIXTURE_PASS"),
    }
    return execution, resumed, _copy_json(difference), publisher, artifact


def _normalize_dry_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    plan = raw.get("plan_audit")
    if type(plan) is not dict:
        plan = raw

    def pick(
        primary: Mapping[str, Any], names: Sequence[str], *, secondary: Mapping[str, Any] = raw
    ) -> Any:
        for source in (primary, secondary):
            for name in names:
                if name in source:
                    return source[name]
        raise ValueError(f"dry-run report lacks aliases: {tuple(names)}")

    condition_raw = pick(plan, ("condition_counts", "condition_snapshot_counts"))
    backend_raw = pick(plan, ("backend_trial_counts",))
    if type(condition_raw) is not dict or type(backend_raw) is not dict:
        raise ValueError("dry-run condition/backend counts are not objects")
    conditions = {
        "IDEAL_MATCHED": condition_raw.get("IDEAL_MATCHED", 0),
        "INDEPENDENT_NOISE_FREE": condition_raw.get(
            "INDEPENDENT_NOISE_FREE", 0
        ),
        "FULL_NOISE": condition_raw.get("FULL_NOISE", 0),
    }
    native_count = backend_raw.get("Native")
    if native_count is None:
        native_count = pick(plan, ("native_trial_count",))
    backends = {
        "Open3D": backend_raw.get(
            "Open3D", backend_raw.get("open3d_point_to_plane", 0)
        ),
        "PCL": backend_raw.get("PCL", backend_raw.get("pcl_point_to_plane", 0)),
        "Native": native_count,
    }
    root_absent = bool(
        raw.get("V3_FORMAL_RUNTIME_ROOT_NOT_CREATED") is True
        or (
            raw.get("formal_root_created") is False
            and raw.get("formal_root_existed_before") is False
            and raw.get("formal_root_existed_after") is False
        )
    )
    return {
        "schema_version": "synthetic_confirmatory_v3_dry_run_report_v1",
        "V3_DRY_RUN_PASS": bool(
            raw.get(
                "V3_DRY_RUN_PASS",
                raw.get("SYNTHETIC_CONFIRMATORY_V3_DRY_RUN_PASS"),
            )
            is True
        ),
        "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": root_absent,
        "planned_snapshot_count": pick(plan, ("planned_snapshot_count",)),
        "unique_snapshot_count": pick(
            plan, ("unique_snapshot_count", "planned_snapshot_unique_count")
        ),
        "planned_trial_count": pick(plan, ("planned_trial_count",)),
        "unique_trial_count": pick(
            plan, ("unique_trial_count", "planned_trial_unique_count")
        ),
        "condition_counts": conditions,
        "backend_trial_counts": backends,
        "duplicate_snapshot_count": pick(plan, ("duplicate_snapshot_count",)),
        "duplicate_trial_count": pick(plan, ("duplicate_trial_count",)),
        "pairing_violation_count": pick(plan, ("pairing_violation_count",)),
        "independent_pseudoreplication_count": pick(
            plan,
            (
                "independent_pseudoreplication_count",
                "independent_pseudoreplication_plan_count",
            ),
        ),
        "V3_RNG_INSTANTIATION_COUNT": pick(
            raw,
            (
                "V3_RNG_INSTANTIATION_COUNT",
                "FORMAL_RNG_INSTANTIATION_COUNT",
                "CONFIRMATORY_RNG_INSTANTIATION_COUNT",
            ),
        ),
        "V3_SNAPSHOT_CONSTRUCTION_COUNT": pick(
            raw,
            (
                "V3_SNAPSHOT_CONSTRUCTION_COUNT",
                "FORMAL_SNAPSHOT_CONSTRUCTION_COUNT",
                "CONFIRMATORY_SNAPSHOT_GENERATION_COUNT",
            ),
        ),
        "V3_BACKEND_EXECUTION_COUNT": pick(
            raw,
            (
                "V3_BACKEND_EXECUTION_COUNT",
                "FORMAL_BACKEND_EXECUTION_COUNT",
                "CONFIRMATORY_BACKEND_EXECUTION_COUNT",
            ),
        ),
        "V3_TRIAL_RESULT_COUNT": pick(
            raw,
            (
                "V3_TRIAL_RESULT_COUNT",
                "FORMAL_TRIAL_RESULT_COUNT",
                "CONFIRMATORY_TRIAL_RESULT_COUNT",
            ),
        ),
        "V3_STARTED_EVENT_COUNT": pick(
            raw,
            (
                "V3_STARTED_EVENT_COUNT",
                "FORMAL_ATTEMPT_EVENT_COUNT",
                "attempt_started_event_count",
            ),
        ),
        "formal_runtime_root": raw.get("formal_runtime_root"),
        "formal_manifest_sha256": raw.get(
            "formal_manifest_sha256", raw.get("manifest_sha256")
        ),
        "source_report_schema_version": raw.get("schema_version"),
    }


def _normalize_suite(value: Mapping[str, Any]) -> dict[str, Any]:
    failures = value.get("failure_count", value.get("failures"))
    errors = value.get("error_count", value.get("errors"))
    unexpected = value.get("unexpected_skip_count", value.get("skipped"))
    return {
        "pass": value.get("pass") is True,
        "test_count": value.get("test_count", value.get("tests")),
        "passed_count": value.get("passed_count", value.get("passed")),
        "failure_count": failures,
        "error_count": errors,
        "unexpected_skip_count": unexpected,
        "junit_path": value.get("junit_path"),
        "junit_sha256": value.get("junit_sha256"),
    }


def _normalize_test_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "v3_specialized": ("v3_specialized",),
        "runtime_lifecycle_specialized": ("runtime_lifecycle_specialized",),
        "v2_scientific_chain_regression": (
            "v2_scientific_chain_regression",
            "v2_specialized",
        ),
        "full_harness": ("full_harness",),
        "pcl_backend_v3": ("pcl_backend_v3",),
    }
    source = raw.get("suites", raw)
    if type(source) is not dict:
        raise ValueError("combined test report lacks suites")
    suites: dict[str, dict[str, Any]] = {}
    for canonical, candidates in aliases.items():
        selected = next((source.get(name) for name in candidates if name in source), None)
        if type(selected) is not dict:
            raise ValueError(f"combined test report lacks suite: {canonical}")
        suites[canonical] = _normalize_suite(selected)
    passed = all(
        suite["pass"]
        and suite["failure_count"] == 0
        and suite["error_count"] == 0
        and suite["unexpected_skip_count"] == 0
        for suite in suites.values()
    )
    return {
        "schema_version": "synthetic_confirmatory_v3_combined_test_report_v1",
        "V3_TEST_SUITE_PASS": bool(
            raw.get("V3_TEST_SUITE_PASS", raw.get("pass")) is True and passed
        ),
        "source_degen_lio_pytest_executed": raw.get(
            "source_degen_lio_pytest_executed", False
        ),
        "suites": suites,
    }


def _implementation_manifest(
    repository: Path, candidate_commit: str, candidate_tag: str
) -> dict[str, Any]:
    changed = _git(
        repository,
        "diff",
        "--name-only",
        f"{RUNTIME_BASE_COMMIT}..{candidate_commit}",
    ).splitlines()
    files = {
        relative: _file_sha256(repository / relative)
        for relative in sorted(filter(None, changed))
        if (repository / relative).is_file()
    }
    return {
        "schema_version": "synthetic_confirmatory_v3_implementation_manifest_v1",
        "V3_IMPLEMENTATION_BINDING_PASS": bool(files),
        "candidate_commit": candidate_commit,
        "candidate_tag": candidate_tag,
        "candidate_branch": "feature/zero-perturbation-synthetic-confirmatory-v3-prerun",
        "runtime_lifecycle_base_commit": RUNTIME_BASE_COMMIT,
        "files": files,
    }


def _formal_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
        V3_EXPECTED_BRANCH,
        V3_EXPECTED_RELEASE_TAG,
        V3_FORMAL_RUNTIME_ROOT,
        V3_RUN_ID,
    )

    serialized = json.dumps(raw, sort_keys=True)
    return {
        **_copy_json(raw),
        "schema_version": "synthetic_confirmatory_v3_formal_execution_profile_v1",
        "FORMAL_EXECUTION_PROFILE_PASS": all(
            token in serialized
            for token in (
                V3_EXPECTED_BRANCH,
                V3_EXPECTED_RELEASE_TAG,
                V3_RUN_ID,
                str(V3_FORMAL_RUNTIME_ROOT),
            )
        ),
        "run_id": V3_RUN_ID,
        "workers": 2,
        "runtime_root": str(V3_FORMAL_RUNTIME_ROOT),
        "expected_branch": V3_EXPECTED_BRANCH,
        "expected_release_tag": V3_EXPECTED_RELEASE_TAG,
        "formal_execution_count": 0,
    }


def _formal_commands() -> str:
    root = "/home/lj/zero_perturbation_phase_a_harness_20260729_1407"
    runtime = "/home/lj/ZPRM/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3"
    manifest = "frozen_assets/synthetic_confirmatory_formal_manifest_v3.json"
    common = (
        "env -u PYTHONPATH PYTHONNOUSERSITE=1 "
        "MAMBA_ROOT_PREFIX=/home/lj/.local/share/degen-lio-micromamba "
        "/home/lj/.local/bin/micromamba run -n degen-lio-zprm-py311 python"
    )
    runner = (
        f"{common} scripts/run_synthetic_confirmatory_v3.py "
        f"--manifest {manifest} --run-id synthetic-confirmatory-v3 "
        f"--runtime-root {runtime} --workers 2 --resume"
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# Frozen commands only; this file was not executed during pre-run qualification.",
            f"cd {root}",
            "git switch feature/zero-perturbation-synthetic-confirmatory-v3-prerun",
            "test \"$(git rev-parse HEAD)\" = \"$(git rev-list -n 1 archive/zero-perturbation-synthetic-confirmatory-v3-pre-run-pass)\"",
            "# Fresh and infrastructure-resume modes intentionally execute the exact same frozen command.",
            "case \"${1:-fresh}\" in",
            f"  fresh) {runner} ;;",
            f"  resume) {runner} ;;",
            "  *) echo 'usage: formal_run_commands.sh [fresh|resume]' >&2; exit 2 ;;",
            "esac",
            f"{common} scripts/analyze_synthetic_confirmatory_v3.py --manifest {manifest} --runtime-root {runtime}",
            f"{common} scripts/verify_synthetic_confirmatory_v3.py --manifest {manifest} --runtime-root {runtime}",
            f"{common} scripts/publish_synthetic_confirmatory_v3.py --manifest {manifest} --runtime-root {runtime}",
            f"{common} scripts/verify_synthetic_confirmatory_v3_artifact.py --manifest {manifest} --runtime-root {runtime}",
            "",
        ]
    )


def _report(decision: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Synthetic Confirmatory v3 — Pre-Run Qualification",
            "",
            "This package contains qualification evidence only. No formal v3 seed was instantiated and no formal snapshot or backend trial was executed.",
            "",
            "- `SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS = true`",
            "- `CONFIRMATORY_V3_RUN_AUTHORIZED = true`",
            "- `SYNTHETIC_CONFIRMATORY_V3_EXECUTED = false`",
            "- `SYNTHETIC_CONFIRMATORY_V3_COMPLETE = false`",
            "- `SYNTHETIC_CONFIRMATORY_V3_PASS = NOT_EVALUATED`",
            "",
            f"- Planned snapshots/trials: `{plan['planned_snapshot_count']} / {plan['planned_trial_count']}`",
            f"- Formal execution count: `{decision['FORMAL_CONFIRMATORY_EXECUTION_COUNT']}`",
            "- Formal runtime root created: `false`",
            "",
        ]
    )


def _write_package(staging: Path, payload: Mapping[str, Any | str]) -> None:
    from phase_a_harness.runtime_lifecycle_io import atomic_create_bytes, canonical_json_bytes
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import PAYLOAD_FILES

    if set(payload) != set(PAYLOAD_FILES):
        raise ValueError("v3 pre-run payload inventory differs from the 28-file contract")
    for name in PAYLOAD_FILES:
        value = payload[name]
        data = value.encode("utf-8") if type(value) is str else canonical_json_bytes(value)
        atomic_create_bytes(staging / name, data)
    rows = []
    for name in sorted(PAYLOAD_FILES):
        path = staging / name
        rows.append((name, path.stat().st_size, _file_sha256(path)))
    manifest = ["relative_path,size_bytes,sha256\n"]
    manifest.extend(f"{name},{size},{digest}\n" for name, size, digest in rows)
    atomic_create_bytes(staging / "MANIFEST.csv", "".join(manifest).encode("utf-8"))
    checksum_paths = sorted({*PAYLOAD_FILES, "MANIFEST.csv"})
    checksum_text = "".join(
        f"{_file_sha256(staging / name)}  {name}\n" for name in checksum_paths
    )
    atomic_create_bytes(staging / "SHA256SUMS", checksum_text.encode("utf-8"))


def build_v3_prerun_artifact(
    repository: str | Path,
    destination: str | Path,
    candidate_commit: str,
    candidate_tag: str,
    adapter_report: str | Path,
    dry_run_report: str | Path,
    test_report: str | Path,
    final_binding: Mapping[str, Any],
    *,
    evidence_dir: str | Path | None = None,
    enforce_external_destination: bool = True,
) -> dict[str, Any]:
    """Build and independently verify the no-clobber compact pre-run package."""

    root = Path(repository).resolve()
    output = Path(os.path.abspath(os.fspath(destination)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"v3 pre-run destination already exists: {output}")
    if enforce_external_destination:
        expected_parent = Path(
            "/home/lj/ZPRM/zero_perturbation_runtime/qualification/"
            "synthetic_confirmatory_v3_prerun"
        )
        if expected_parent not in output.parents:
            raise ValueError("v3 compact pre-run destination is outside qualification root")
    if _git(root, "rev-parse", "HEAD") != candidate_commit:
        raise ValueError("candidate commit does not equal HEAD")
    if _git(root, "branch", "--show-current") != "feature/zero-perturbation-synthetic-confirmatory-v3-prerun":
        raise ValueError("unexpected v3 pre-run branch")
    if _git(root, "rev-list", "-n", "1", candidate_tag) != candidate_commit:
        raise ValueError("candidate tag does not peel to candidate commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("candidate worktree is not clean")
    evidence_root = Path(evidence_dir).resolve() if evidence_dir else None
    paths = _standard_paths(root)
    adapter = _strict_object(adapter_report)
    dry = _normalize_dry_run(_strict_object(dry_run_report))
    tests = _normalize_test_report(_strict_object(test_report))
    runtime = _runtime_lifecycle_binding(root)
    v1, v2, retired = _failure_and_retirement_bindings(root)
    design = _v3_design_audit(root, paths)
    from phase_a_harness.runtime_lifecycle_science_audit import scientific_core_binding
    science = scientific_core_binding(root)
    science["SCIENTIFIC_CORE_BINDING_PASS"] = bool(
        science.get("SCIENTIFIC_CORE_FILE_CHANGE_COUNT") == 0
        and science.get("H1_H6_SEMANTICS_CHANGE_COUNT") == 0
        and science.get("FROZEN_MODEL_CHANGE_COUNT") == 0
        and science.get("BACKEND_BINDING_CHANGE_COUNT") == 0
    )
    lifecycle = _runtime_lifecycle_core_binding(root)
    schedule = _strict_object(paths["schedule"])
    provenance = _seed_provenance(evidence_root, schedule)
    plan = _plan_audit(paths)
    scientific_diff = _scientific_diff(root, evidence_root)
    model_sha = _file_sha256(paths["model"])
    model = {
        "schema_version": "synthetic_confirmatory_v3_frozen_model_binding_v1",
        "FROZEN_MODEL_SHA_MATCH": model_sha
        == "99806f83d3a0d2393ee7e36e8c06f14a1a8a44fb8d02b074ebc2d9fe750d0872",
        "path": paths["model"].relative_to(root).as_posix(),
        "sha256": model_sha,
    }
    backend = _backend_binding(root)
    profile = _formal_profile(_strict_object(paths["profile"]))
    path_contract = _runtime_path_contract(root, profile)
    fixture, resume, primary, publisher, fixture_artifact = _fixture_evidence(adapter)
    implementation = _implementation_manifest(root, candidate_commit, candidate_tag)
    binding = _copy_json(final_binding)
    if binding.get("V3_FINAL_GIT_BINDING_PASS") is not True:
        raise ValueError("stage-A/final-transition binding did not pass")
    gates = (
        runtime.get("RUNTIME_LIFECYCLE_QUALIFICATION_BINDING_PASS") is True,
        v1.get("V1_FAILURE_HISTORY_PRESERVED") is True,
        v2.get("V2_FAILURE_HISTORY_PRESERVED") is True,
        retired.get("RETIRED_SEED_SETS_PASS") is True,
        design.get("V3_DESIGN_AUDIT_PASS") is True,
        science.get("SCIENTIFIC_CORE_BINDING_PASS") is True,
        lifecycle.get("RUNTIME_LIFECYCLE_CORE_BINDING_PASS") is True,
        path_contract.get("V3_RUNTIME_PATH_POLICY_PASS") is True,
        adapter.get("V3_EXECUTION_ADAPTER_FIXTURE_PASS") is True,
        scientific_diff.get("V2_TO_V3_SCIENTIFIC_DIFF_PASS") is True,
        model.get("FROZEN_MODEL_SHA_MATCH") is True,
        backend.get("BACKEND_BINDING_MATCH") is True,
        provenance.get("NEW_V3_NAMESPACE_COLLISION") is False,
        provenance.get("V3_SEED_PROVENANCE_COLLISION_COUNT") == 0,
        plan.get("V3_PLAN_PASS") is True,
        dry.get("V3_DRY_RUN_PASS") is True,
        tests.get("V3_TEST_SUITE_PASS") is True,
        binding.get("V3_FINAL_GIT_BINDING_PASS") is True,
    )
    if not all(gates):
        raise PermissionError("v3 pre-run gate is not fully qualified")
    decision = {
        "schema_version": "synthetic_confirmatory_v3_prerun_decision_v1",
        "V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS": True,
        "V3_FINAL_GIT_BINDING_PASS": True,
        "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS": True,
        "CONFIRMATORY_V3_RUN_AUTHORIZED": True,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "FORMAL_CONFIRMATORY_EXECUTION_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
    }
    run_manifest = {
        "schema_version": "synthetic_confirmatory_v3_prerun_run_manifest_v1",
        "run_id": "synthetic-confirmatory-v3",
        "qualification_candidate_commit": candidate_commit,
        "qualification_candidate_tag": candidate_tag,
        "planned_snapshot_count": 595,
        "planned_trial_count": 1190,
        "formal_snapshot_construction_count": 0,
        "formal_backend_execution_count": 0,
        "formal_trial_result_count": 0,
        "native_execution_count": 0,
        "adapter_report_path": str(Path(adapter_report).resolve()),
        "adapter_report_sha256": _file_sha256(adapter_report),
        "dry_run_report_path": str(Path(dry_run_report).resolve()),
        "dry_run_report_sha256": _file_sha256(dry_run_report),
        "test_report_path": str(Path(test_report).resolve()),
        "test_report_sha256": _file_sha256(test_report),
    }
    payload: dict[str, Any | str] = {
        "runtime_lifecycle_qualification_binding.json": runtime,
        "v1_failure_binding.json": v1,
        "v2_failure_binding.json": v2,
        "retired_seed_sets.json": retired,
        "v3_design_audit.json": design,
        "scientific_core_binding.json": science,
        "runtime_lifecycle_core_binding.json": lifecycle,
        "runtime_path_contract.json": path_contract,
        "v3_seed_schedule.json": schedule,
        "v3_seed_provenance_audit.json": provenance,
        "v3_plan_audit.json": plan,
        "v2_to_v3_scientific_diff.json": scientific_diff,
        "frozen_model_binding.json": model,
        "backend_binding.json": backend,
        "fixture_execution_report.json": fixture,
        "fixture_resume_report.json": resume,
        "primary_independent_difference.json": primary,
        "publisher_inventory.json": publisher,
        "fixture_artifact_verification.json": fixture_artifact,
        "v3_dry_run_report.json": dry,
        "test_report.json": tests,
        "implementation_manifest.json": implementation,
        "formal_execution_profile.json": profile,
        "formal_run_commands.sh": _formal_commands(),
        "final_binding_audit.json": binding,
        "final_decision.json": decision,
        "run_manifest.json": run_manifest,
        "pre_run_report.md": _report(decision, plan),
    }
    from phase_a_harness.runtime_lifecycle_io import atomic_publish_directory
    from phase_a_harness.synthetic_confirmatory_v3_artifact_verifier import (
        verify_v3_prerun_artifact,
    )

    verification: dict[str, Any] = {}

    def populate(staging: Path) -> None:
        nonlocal verification
        _write_package(staging, payload)
        verification = verify_v3_prerun_artifact(staging)
        if verification.get("V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS") is not True:
            raise ValueError(
                "assembled v3 pre-run artifact failed independent verification: "
                f"{verification.get('evidence_semantic_failures')}"
            )

    atomic_publish_directory(output, populate)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("external pre-run publication polluted the Git worktree")
    read_only = verify_v3_prerun_artifact(output)
    if read_only != verification:
        raise RuntimeError("published v3 pre-run verification differs from staging")
    return {**read_only, "artifact_path": str(output)}


def _stage_a_final_binding(
    repository: Path, candidate_commit: str, candidate_tag: str
) -> dict[str, Any]:
    clean = not _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "schema_version": "synthetic_confirmatory_v3_final_binding_audit_v1",
        "V3_FINAL_GIT_BINDING_PASS": bool(
            clean
            and _git(repository, "rev-parse", "HEAD") == candidate_commit
            and _git(repository, "rev-list", "-n", "1", candidate_tag)
            == candidate_commit
        ),
        "binding_stage": "STAGE_A_CANDIDATE_WITH_ARTIFACT_ONLY_FINAL_TRANSITION",
        "candidate_commit": candidate_commit,
        "candidate_tag": candidate_tag,
        "candidate_worktree_clean": clean,
        "allowed_final_diff_prefix": "artifacts/synthetic_confirmatory_v3_prerun/",
        "final_tag_requires_post_import_read_only_gate": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-candidate-commit", required=True)
    parser.add_argument("--expected-candidate-tag", required=True)
    parser.add_argument("--adapter-report", type=Path, default=DEFAULT_ADAPTER_REPORT)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--final-binding-report", type=Path)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    _assert_environment(repository)
    sys.path.insert(0, str(repository / "src"))
    final_binding = (
        _strict_object(args.final_binding_report)
        if args.final_binding_report
        else _stage_a_final_binding(
            repository,
            args.expected_candidate_commit,
            args.expected_candidate_tag,
        )
    )
    report = build_v3_prerun_artifact(
        repository,
        args.destination,
        args.expected_candidate_commit,
        args.expected_candidate_tag,
        args.adapter_report,
        args.dry_run_report,
        args.test_report,
        final_binding,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("V3_PRE_RUN_ARTIFACT_VERIFICATION_PASS") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
