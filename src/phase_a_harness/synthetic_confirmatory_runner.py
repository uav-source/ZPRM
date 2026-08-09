"""Dry-run qualification and frozen formal runner for Confirmatory v1.

``dry_run_synthetic_confirmatory`` is intentionally metadata-only: it reads the
signed manifest, frozen protocol assets, and the two plan CSVs.  It never
creates the output directory, opens a snapshot cache, constructs an RNG,
imports a backend adapter, or writes an attempt event.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256
from .synthetic_confirmatory_manifest import (
    FORMAL_OUTPUT_DIR,
    FORMAL_RUN_ID,
    FORMAL_WORKERS,
    MANIFEST_RELATIVE,
    RAW_RESULT_MANIFEST_SCHEMA,
    verify_synthetic_confirmatory_manifest,
)
from .synthetic_confirmatory_snapshot_builder import (
    BACKENDS,
    CONDITIONS,
    SNAPSHOT_COUNT,
    SNAPSHOT_LOCK_RELATIVE,
    TRIAL_COUNT,
)


DRY_RUN_SCHEMA = "synthetic_confirmatory_dry_run_v1"
FORMAL_RUN_SCHEMA = "synthetic_confirmatory_formal_run_v1"
PRERUN_ARTIFACT_RELATIVE = Path("artifacts/synthetic_confirmatory_prerun_v1")
FORMAL_BRANCH = "feature/zero-perturbation-synthetic-confirmatory-prerun"
FORMAL_PRERUN_TAG = "archive/zero-perturbation-synthetic-confirmatory-pre-run-pass"
SNAPSHOT_FIELDS = (
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "planned_backend_count",
    "replicate_semantics",
)
TRIAL_FIELDS = (
    "planned_trial_id",
    "planned_snapshot_id",
    "scene_variant",
    "condition",
    "geometry_seed",
    "measurement_seed",
    "repeat_index",
    "backend",
)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key in {label}: {name}")
            result[name] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ValueError(f"unexpected CSV schema: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid plan CSV: {path}") from error
    if any(None in row for row in rows):
        raise ValueError(f"CSV row has excess columns: {path}")
    return rows


def _integer(value: str, name: str) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{name} is not a canonical integer")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer") from error
    if str(parsed) != value:
        raise ValueError(f"{name} is not a canonical integer")
    return parsed


def _optional_integer(value: str, name: str) -> int | None:
    return None if value == "" else _integer(value, name)


def _plan_identity_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _typed_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": _optional_integer(
                row["measurement_seed"], "measurement_seed"
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "planned_backend_count": _integer(
                row["planned_backend_count"], "planned_backend_count"
            ),
            "replicate_semantics": row["replicate_semantics"],
        }
        for row in _read_csv(path, SNAPSHOT_FIELDS)
    ]


def _typed_trial_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "planned_trial_id": row["planned_trial_id"],
            "planned_snapshot_id": row["planned_snapshot_id"],
            "scene_variant": row["scene_variant"],
            "condition": row["condition"],
            "geometry_seed": _integer(row["geometry_seed"], "geometry_seed"),
            "measurement_seed": _optional_integer(
                row["measurement_seed"], "measurement_seed"
            ),
            "repeat_index": _integer(row["repeat_index"], "repeat_index"),
            "backend": row["backend"],
        }
        for row in _read_csv(path, TRIAL_FIELDS)
    ]


def _repository_local(root: Path, relative: Any, label: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError(f"invalid repository-relative {label}")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"{label} escaped repository")
    return candidate


def verify_formal_prerun_artifact_binding(
    root: str | Path,
    manifest_file: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the exact pre-run artifact against the authorized manifest."""

    repository = Path(root).resolve()
    manifest_path = Path(manifest_file).resolve()
    if manifest_path != (repository / MANIFEST_RELATIVE).resolve():
        raise ValueError("formal manifest path is not canonical")
    if manifest.get("formal_execution_authorized") is not True:
        raise PermissionError("pre-run artifact cannot authorize a false manifest")
    artifact = (repository / PRERUN_ARTIFACT_RELATIVE).resolve()
    if not artifact.is_dir():
        raise FileNotFoundError("exact Synthetic Confirmatory pre-run artifact is missing")

    from .synthetic_confirmatory_artifact_verifier import (
        verify_synthetic_confirmatory_prerun_artifact,
    )

    live = verify_synthetic_confirmatory_prerun_artifact(
        artifact, write_report=False
    )
    recorded = _load_json_object(
        artifact / "artifact_verification.json",
        "recorded pre-run artifact verification",
    )
    decision = _load_json_object(
        artifact / "final_decision.json", "pre-run final decision"
    )
    implementation = _load_json_object(
        artifact / "implementation_manifest.json",
        "pre-run implementation manifest",
    )
    run = _load_json_object(
        artifact / "run_manifest.json", "pre-run run manifest"
    )
    manifest_file_sha = file_sha256(manifest_path)
    decision_pass = bool(
        decision.get("SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS") is True
        and decision.get("CONFIRMATORY_RUN_AUTHORIZED") is True
        and decision.get("SYNTHETIC_CONFIRMATORY_EXECUTED") is False
        and decision.get("SYNTHETIC_CONFIRMATORY_COMPLETE") is False
        and decision.get("SYNTHETIC_CONFIRMATORY_PASS") == "NOT_EVALUATED"
        and decision.get("REAL_DATA_RUN_AUTHORIZED") is False
        and decision.get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED") is False
    )
    implementation_pass = bool(
        implementation.get("schema_version")
        == "synthetic_confirmatory_prerun_implementation_manifest_v1"
        and implementation.get("formal_execution_authorized") is True
        and implementation.get("formal_manifest_path")
        == MANIFEST_RELATIVE.as_posix()
        and implementation.get("formal_manifest_payload_sha256")
        == manifest.get("manifest_payload_sha256")
        and implementation.get("formal_manifest_file_sha256")
        == manifest_file_sha
        and implementation.get("bound_files") == manifest.get("bound_files")
        and implementation.get("formal_run_id") == FORMAL_RUN_ID
        and implementation.get("formal_output_dir") == FORMAL_OUTPUT_DIR
        and implementation.get("formal_workers") == FORMAL_WORKERS
        and implementation.get("formal_branch") == FORMAL_BRANCH
        and implementation.get("formal_pre_run_tag") == FORMAL_PRERUN_TAG
    )
    run_binding_pass = bool(
        run.get("formal_manifest_payload_sha256")
        == manifest.get("manifest_payload_sha256")
        and run.get("formal_run_id") == FORMAL_RUN_ID
        and run.get("final_decision") == decision
    )
    live_pass = bool(
        live.get("CONFIRMATORY_ARTIFACT_VERIFICATION_PASS") is True
        and recorded == live
    )
    passed = bool(
        live_pass and decision_pass and implementation_pass and run_binding_pass
    )
    report = {
        "FORMAL_PRERUN_ARTIFACT_BINDING_PASS": passed,
        "artifact_path": str(artifact),
        "decision_authorization_match_pass": decision_pass,
        "implementation_manifest_binding_pass": implementation_pass,
        "live_artifact_verification_match_pass": live_pass,
        "manifest_file_sha256": manifest_file_sha,
        "manifest_payload_sha256": manifest.get("manifest_payload_sha256"),
        "run_manifest_binding_pass": run_binding_pass,
    }
    if not passed:
        raise PermissionError(
            "formal pre-run artifact/manifest authorization binding failed"
        )
    return report


def _git_text(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git gate command failed: {' '.join(arguments)}") from error
    return completed.stdout.strip()


def verify_formal_git_gate(
    root: str | Path, manifest_file: str | Path
) -> dict[str, Any]:
    """Require the committed manifest blob at clean tagged HEAD.

    The tag name is frozen in this manifest-bound runner and in the pre-run
    implementation artifact.  The future commit hash is deliberately not
    embedded in the manifest, avoiding a commit/manifest self-reference.
    """

    repository = Path(root).resolve()
    manifest_path = Path(manifest_file).resolve()
    if manifest_path != (repository / MANIFEST_RELATIVE).resolve():
        raise ValueError("formal manifest path is not canonical")
    top = Path(_git_text(repository, "rev-parse", "--show-toplevel")).resolve()
    head = _git_text(repository, "rev-parse", "HEAD^{commit}")
    tag_head = _git_text(
        repository, "rev-parse", f"{FORMAL_PRERUN_TAG}^{{commit}}"
    )
    branch = _git_text(repository, "branch", "--show-current")
    status = _git_text(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    _git_text(
        repository,
        "ls-files",
        "--error-unmatch",
        MANIFEST_RELATIVE.as_posix(),
    )
    try:
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"HEAD:{MANIFEST_RELATIVE.as_posix()}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot read committed formal manifest blob") from error
    blob_match = committed == manifest_path.read_bytes()
    passed = bool(
        top == repository
        and head == tag_head
        and branch == FORMAL_BRANCH
        and not status
        and blob_match
    )
    report = {
        "FORMAL_GIT_GATE_PASS": passed,
        "branch": branch,
        "head_commit": head,
        "manifest_head_blob_match": blob_match,
        "pre_run_tag": FORMAL_PRERUN_TAG,
        "pre_run_tag_commit": tag_head,
        "repository_root": str(top),
        "worktree_clean": not status,
    }
    if not passed:
        raise PermissionError("formal pre-run tag/HEAD/clean git gate failed")
    return report


def _require_protocol_audit(
    report: Mapping[str, Any], *, require_live_seed_unused: bool
) -> None:
    plan = report.get("plan_audit")
    gate = report.get("gate_contract_audit")
    seed = report.get("seed_provenance_audit")
    if not all(type(item) is dict for item in (plan, gate, seed)):
        raise ValueError("Confirmatory protocol audit report is incomplete")
    required = [
        report.get("SCIENTIFIC_SURVIVAL_BINDING_PASS") is True,
        report.get("CONFIRMATORY_PROTOCOL_BINDING_PASS") is True,
        gate.get("CONFIRMATORY_GATE_CONTRACT_PASS") is True,
        seed.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True,
        plan.get("CONFIRMATORY_PLAN_COUNT_PASS") is True,
        plan.get("CONFIRMATORY_PLAN_UNIQUENESS_PASS") is True,
        plan.get("CONFIRMATORY_PLAN_PAIRING_PASS") is True,
        plan.get("CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS") is True,
    ]
    if require_live_seed_unused:
        required.extend(
            (
                report.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS")
                is True,
                seed.get("CONFIRMATORY_SEED_USAGE_HIT_COUNT") == 0,
                seed.get("CONFIRMATORY_SEED_INSTANTIATION_COUNT") == 0,
                seed.get("CONFIRMATORY_SEED_PARSE_ERROR_COUNT") == 0,
                seed.get("CONFIRMATORY_RNG_INSTANTIATION_COUNT") == 0,
            )
        )
    if not all(required):
        raise PermissionError("Synthetic Confirmatory protocol qualification failed")


def _audit_frozen_protocol_for_formal(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate frozen pre-run evidence without scanning legitimate formal results."""

    from .synthetic_confirmatory_protocol import (
        BACKENDS as PROTOCOL_BACKENDS,
        BOOTSTRAP_SEED,
        CONDITIONS as PROTOCOL_CONDITIONS,
        GEOMETRY_SEEDS,
        MEASUREMENT_SEEDS,
        SCENES,
        audit_confirmatory_gate_contract,
        audit_confirmatory_plan,
    )

    plan = audit_confirmatory_plan(
        root / manifest["planned_snapshots_path"],
        root / manifest["planned_trials_path"],
    )
    gate = audit_confirmatory_gate_contract(root / manifest["gate_contract_path"])
    stored_seed = _load_json_object(
        root / manifest["seed_provenance_audit_path"],
        "frozen seed provenance audit",
    )
    seed_pass = bool(
        stored_seed.get("CONFIRMATORY_SEED_PROVENANCE_PASS") is True
        and stored_seed.get("confirmatory_seed_instantiation_count") == 0
        and stored_seed.get("parse_failure_count") == 0
        and stored_seed.get("structured_usage_hits") == []
    )
    protocol = _load_json_object(
        root / manifest["scientific_protocol_path"],
        "frozen Confirmatory protocol",
    )
    unsigned_protocol = {
        name: value
        for name, value in protocol.items()
        if name != "protocol_payload_sha256"
    }
    protocol_pass = bool(
        protocol.get("schema_version") == "synthetic_confirmatory_protocol_v1"
        and protocol.get("protocol_payload_sha256")
        == _plan_identity_sha256(unsigned_protocol)
        and protocol.get("SYNTHETIC_CONFIRMATORY_PROTOCOL_READY") is True
        and protocol.get("CONFIRMATORY_RUN_AUTHORIZED") is False
        and protocol.get("scientific_survival_audit_pass") is True
        and protocol.get("confirmatory_seed_provenance_pass") is True
        and protocol.get("backends") == list(PROTOCOL_BACKENDS)
        and protocol.get("conditions") == list(PROTOCOL_CONDITIONS)
        and protocol.get("scenes") == list(SCENES)
        and protocol.get("geometry_seeds") == list(GEOMETRY_SEEDS)
        and protocol.get("measurement_seeds") == list(MEASUREMENT_SEEDS)
        and protocol.get("bootstrap_seed") == BOOTSTRAP_SEED
        and protocol.get("planned_snapshot_count") == SNAPSHOT_COUNT
        and protocol.get("planned_trial_count") == TRIAL_COUNT
        and protocol.get("planned_snapshot_identity_sha256")
        == plan.get("planned_snapshot_identity_sha256")
        and protocol.get("planned_trial_identity_sha256")
        == plan.get("planned_trial_identity_sha256")
        and protocol.get("gate_contract_payload_sha256")
        == gate.get("recorded_gate_contract_payload_sha256")
        and protocol.get("native_trial_count") == 0
        and protocol.get("registration_execution_count") == 0
        and protocol.get("snapshot_generation_count") == 0
    )
    return {
        "CONFIRMATORY_PROTOCOL_BINDING_PASS": protocol_pass,
        "SCIENTIFIC_SURVIVAL_BINDING_PASS": bool(
            manifest.get("scientific_survival_commit")
            == "ffc15334f4ded25fdba5e709b45657dbad481dfc"
            and manifest.get("scientific_survival_tag")
            == "archive/zero-perturbation-scientific-survival-audit-v1"
        ),
        "SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS": bool(
            protocol_pass
            and seed_pass
            and gate.get("CONFIRMATORY_GATE_CONTRACT_PASS") is True
            and all(
                plan.get(name) is True
                for name in (
                    "CONFIRMATORY_PLAN_COUNT_PASS",
                    "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
                    "CONFIRMATORY_PLAN_PAIRING_PASS",
                    "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
                )
            )
        ),
        "gate_contract_audit": gate,
        "plan_audit": plan,
        "seed_provenance_audit": {
            "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
            "CONFIRMATORY_SEED_PARSE_ERROR_COUNT": 0,
            "CONFIRMATORY_SEED_PROVENANCE_PASS": seed_pass,
            "CONFIRMATORY_SEED_USAGE_HIT_COUNT": 0,
            "evidence_scope": "FROZEN_PRE_RUN_PROVENANCE",
        },
    }


def load_synthetic_confirmatory_stack(
    manifest_path: str | Path, require_authorized: bool
) -> dict[str, Any]:
    """Load the one exact manifest and all read-only plan/parameter contracts."""

    if type(require_authorized) is not bool:
        raise TypeError("require_authorized must be bool")
    manifest_file = Path(manifest_path).resolve()
    if manifest_file.parent.name != "frozen_assets":
        raise ValueError("Confirmatory manifest must reside in frozen_assets")
    root = manifest_file.parent.parent.resolve()
    if manifest_file != (root / MANIFEST_RELATIVE).resolve():
        raise ValueError("Synthetic Confirmatory must use its one v1 manifest")
    candidates = sorted(
        path.resolve()
        for path in manifest_file.parent.glob(
            "synthetic_confirmatory_formal_manifest*.json"
        )
        if path.is_file()
    )
    if candidates != [manifest_file]:
        raise ValueError("multiple or ambiguous Confirmatory manifests exist")
    manifest = verify_synthetic_confirmatory_manifest(
        root, require_authorized=require_authorized
    )

    if require_authorized:
        prerun_artifact_binding = verify_formal_prerun_artifact_binding(
            root, manifest_file, manifest
        )
        formal_git_gate = verify_formal_git_gate(root, manifest_file)
        protocol_audit = _audit_frozen_protocol_for_formal(root, manifest)
    else:
        prerun_artifact_binding = None
        formal_git_gate = None
        from .synthetic_confirmatory_protocol import (
            audit_confirmatory_protocol_contract,
        )

        protocol_audit = audit_confirmatory_protocol_contract(root)
    _require_protocol_audit(
        protocol_audit, require_live_seed_unused=not require_authorized
    )
    snapshot_plan_path = _repository_local(
        root, manifest["planned_snapshots_path"], "planned snapshots path"
    )
    trial_plan_path = _repository_local(
        root, manifest["planned_trials_path"], "planned trials path"
    )
    snapshots = _typed_snapshot_rows(snapshot_plan_path)
    trials = _typed_trial_rows(trial_plan_path)
    plan_audit = protocol_audit["plan_audit"]
    if (
        len(snapshots) != SNAPSHOT_COUNT
        or len(trials) != TRIAL_COUNT
        or plan_audit.get("planned_snapshot_identity_sha256")
        != _plan_identity_sha256(snapshots)
        or plan_audit.get("planned_trial_identity_sha256")
        != _plan_identity_sha256(trials)
    ):
        raise ValueError("runner plan parse differs from audited plan")
    parameters_path = _repository_local(
        root,
        manifest["bound_files"]["backend_parameter_contract"]["path"],
        "backend parameter contract",
    )
    parameter_contract = _load_json_object(
        parameters_path, "backend parameter contract"
    )
    open3d = parameter_contract.get("open3d")
    pcl = parameter_contract.get("pcl")
    if (
        type(open3d) is not dict
        or type(pcl) is not dict
        or canonical_json_sha256(open3d.get("parameters", {}))
        != manifest["open3d_parameter_sha256"]
        or canonical_json_sha256(pcl.get("parameters", {}))
        != manifest["pcl_parameter_sha256"]
    ):
        raise ValueError("backend parameter contract changed")
    cache_root = _repository_local(
        root, manifest["snapshot_cache_root"], "snapshot cache root"
    )
    return {
        "cache_root": cache_root,
        "lock_path": (root / SNAPSHOT_LOCK_RELATIVE).resolve(),
        "manifest": manifest,
        "manifest_file": manifest_file,
        "parameters": {
            "open3d_point_to_plane": open3d["parameters"],
            "pcl_point_to_plane": pcl["parameters"],
        },
        "formal_git_gate": formal_git_gate,
        "pcl_cli": _repository_local(
            root, manifest["bound_files"]["pcl_cli"]["path"], "PCL CLI"
        ),
        "plan_audit": plan_audit,
        "prerun_artifact_binding": prerun_artifact_binding,
        "protocol_audit": protocol_audit,
        "root": root,
        "snapshots": snapshots,
        "trials": trials,
    }


def _assert_invocation(
    stack: Mapping[str, Any],
    *,
    run_id: str,
    output_dir: str | Path,
    workers: int,
) -> Path:
    manifest = stack["manifest"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    output = Path(output_dir).resolve()
    expected_output = (stack["root"] / manifest["formal_output_dir"]).resolve()
    if (
        run_id != manifest["formal_run_id"]
        or workers != manifest["formal_workers"]
        or output != expected_output
        or manifest["formal_run_id"] != FORMAL_RUN_ID
        or manifest["formal_output_dir"] != FORMAL_OUTPUT_DIR
        or manifest["formal_workers"] != FORMAL_WORKERS
    ):
        raise ValueError("Confirmatory invocation differs from formal manifest")
    return output


def dry_run_synthetic_confirmatory(
    *, manifest_path: str | Path, run_id: str, output_dir: str | Path, workers: int
) -> dict[str, Any]:
    """Validate the full design while proving that every execution count is zero."""

    stack = load_synthetic_confirmatory_stack(
        manifest_path, require_authorized=False
    )
    manifest = stack["manifest"]
    if manifest.get("formal_execution_authorized") is not False:
        raise PermissionError("Confirmatory dry-run requires authorization=false")
    output = _assert_invocation(
        stack, run_id=run_id, output_dir=output_dir, workers=workers
    )
    if output.exists():
        raise FileExistsError("Confirmatory dry-run requires an absent output directory")
    snapshot_counts = Counter(row["condition"] for row in stack["snapshots"])
    backend_counts = Counter(row["backend"] for row in stack["trials"])
    plan = stack["plan_audit"]
    expected_snapshot_counts = {
        "IDEAL_MATCHED": 35,
        "INDEPENDENT_NOISE_FREE": 35,
        "FULL_NOISE": 525,
    }
    plan_gates = (
        "CONFIRMATORY_PLAN_COUNT_PASS",
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS",
        "CONFIRMATORY_PLAN_PAIRING_PASS",
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS",
    )
    passed = bool(
        len(stack["snapshots"]) == SNAPSHOT_COUNT
        and len(stack["trials"]) == TRIAL_COUNT
        and dict(snapshot_counts) == expected_snapshot_counts
        and backend_counts
        == Counter(
            {"open3d_point_to_plane": 595, "pcl_point_to_plane": 595}
        )
        and all(plan.get(name) is True for name in plan_gates)
        and plan.get("native_trial_count") == 0
        and not output.exists()
    )
    return {
        "CONFIRMATORY_BACKEND_EXECUTION_COUNT": 0,
        "CONFIRMATORY_DRY_RUN_PASS": passed,
        "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
        "CONFIRMATORY_SNAPSHOT_GENERATION_COUNT": 0,
        "CONFIRMATORY_TRIAL_RESULT_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
        "attempt_started_event_count": 0,
        "confirmatory_backend_execution_count": 0,
        "confirmatory_rng_instantiation_count": 0,
        "confirmatory_snapshot_generation_count": 0,
        "confirmatory_trial_result_count": 0,
        "condition_snapshot_counts": expected_snapshot_counts,
        "duplicate_snapshot_count": plan["duplicate_snapshot_count"],
        "duplicate_trial_count": plan["duplicate_trial_count"],
        "formal_execution_authorized_before_freeze": False,
        "independent_pseudoreplication_plan_count": plan[
            "independent_pseudoreplication_plan_count"
        ],
        "native_trial_count": 0,
        "open3d_trial_count": backend_counts["open3d_point_to_plane"],
        "output_dir": str(output),
        "output_dir_created": False,
        "pairing_violation_count": plan["pairing_violation_count"],
        "pcl_trial_count": backend_counts["pcl_point_to_plane"],
        "planned_snapshot_count": len(stack["snapshots"]),
        "planned_trial_count": len(stack["trials"]),
        "run_id": run_id,
        "schema_version": DRY_RUN_SCHEMA,
        "workers": workers,
    }


def _fixture(stack: Mapping[str, Any], row: Mapping[str, Any]) -> Any:
    import numpy as np

    from .phase_a_execution_chain_fixture import FixtureSnapshot
    from .synthetic_confirmatory_snapshot_builder import (
        read_synthetic_confirmatory_snapshot,
    )

    snapshot_id = row["planned_snapshot_id"]
    lock_entry = stack["lock_by_id"][snapshot_id]
    item = read_synthetic_confirmatory_snapshot(
        stack["cache_root"], row, expected_lock_entry=lock_entry, arrays=True
    )
    metadata = item["metadata"]
    return FixtureSnapshot(
        snapshot_id=snapshot_id,
        scene_variant=row["scene_variant"],
        condition=row["condition"],
        source=np.asarray(item["source"]),
        target=np.asarray(item["target"]),
        reference=np.asarray(item["reference"]),
        expected_failure_classifications=("NONE",),
        checksums={
            "source_checksum": metadata["source_checksum"],
            "target_checksum": metadata["target_checksum"],
            "reference_pose_checksum": metadata["reference_pose_checksum"],
            "snapshot_checksum": metadata["snapshot_checksum"],
        },
    )


def _common(
    stack: Mapping[str, Any], fixture: Any, row: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "backend": row["backend"],
        "condition": fixture.condition,
        "implementation_sha256": stack["manifest"]["manifest_payload_sha256"],
        "planned_trial_id": row["planned_trial_id"],
        "protocol_sha256": stack["manifest"]["scientific_protocol_sha256"],
        "reference_pose_checksum": fixture.checksums["reference_pose_checksum"],
        "scene_variant": fixture.scene_variant,
        "schema_version": "phase_a_trial_result_v1",
        "snapshot_checksum": fixture.checksums["snapshot_checksum"],
        "snapshot_id": fixture.snapshot_id,
        "snapshot_lock_sha256": stack["snapshot_lock_sha256"],
        "source_checksum": fixture.checksums["source_checksum"],
        "target_checksum": fixture.checksums["target_checksum"],
    }


def _validate_confirmatory_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("condition") == "IDEAL_MATCHED":
        from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict

        return validate_phase_a_trial_result_strict(value)
    from .full_synthetic_trial_result import (
        validate_full_synthetic_trial_result_strict,
    )

    return validate_full_synthetic_trial_result_strict(value)


def _execute_one(
    stack: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _fixture(stack, row)
    common = _common(stack, fixture, row)
    condition = row["condition"]
    backend = row["backend"]
    if condition == "IDEAL_MATCHED":
        from .phase_a_execution_chain_audit import (
            execute_open3d_fixture,
            execute_pcl_fixture,
        )

        if backend == "open3d_point_to_plane":
            result = execute_open3d_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
            )
        elif backend == "pcl_point_to_plane":
            result = execute_pcl_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
                pcl_cli=stack["pcl_cli"],
            )
        else:
            raise PermissionError("Native and unknown backends are forbidden")
    else:
        from .full_synthetic_backend_execution import (
            execute_full_synthetic_open3d_fixture,
            execute_full_synthetic_pcl_fixture,
        )

        if backend == "open3d_point_to_plane":
            result = execute_full_synthetic_open3d_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
            )
        elif backend == "pcl_point_to_plane":
            result = execute_full_synthetic_pcl_fixture(
                fixture=fixture,
                common=common,
                parameters=stack["parameters"][backend],
                pcl_cli=stack["pcl_cli"],
            )
        else:
            raise PermissionError("Native and unknown backends are forbidden")
    return common, _validate_confirmatory_result(result)


def _empty_raw_manifest(run_id: str) -> dict[str, Any]:
    return {
        "results": {},
        "run_id": run_id,
        "schema_version": RAW_RESULT_MANIFEST_SCHEMA,
    }


def _read_raw_manifest(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        return _empty_raw_manifest(run_id)
    value = _load_json_object(path, "Confirmatory raw result manifest")
    if (
        set(value) != {"results", "run_id", "schema_version"}
        or value["run_id"] != run_id
        or value["schema_version"] != RAW_RESULT_MANIFEST_SCHEMA
        or type(value["results"]) is not dict
    ):
        raise ValueError("Confirmatory raw result manifest identity changed")
    return value


def _recover_atomic_result_orphans(
    stack: Mapping[str, Any],
    results_dir: Path,
    raw_manifest_path: Path,
    raw_manifest: dict[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Adopt only canonical, fully validated results from the write/manifest gap.

    A result is written atomically before its raw-manifest entry.  Process loss
    in that narrow interval leaves one legitimate orphan.  Recovery derives the
    only possible trial ID from the frozen result filename, validates all 26
    fields and expected identity/checksum bindings, requires byte-canonical
    JSON, and only then atomically records its freshly computed SHA-256.
    """

    from .phase_a_trial_result_schema import canonical_json_bytes
    from .phase_a_trial_result_writer import atomic_write_bytes, result_filename

    if type(raw_manifest) is not dict or type(raw_manifest.get("results")) is not dict:
        raise ValueError("raw result manifest is not mutable canonical state")
    # First prove all manifest-referenced results are valid.  Unreferenced files
    # are considered below, but missing/corrupt/extra manifest entries fail
    # before recovery can mutate the manifest.
    _audit_raw_inventory(
        stack,
        results_dir,
        raw_manifest,
        expected_rows,
        allow_unreferenced_files=True,
    )
    if not results_dir.exists():
        return {
            "recovered_orphan_result_count": 0,
            "recovered_orphan_trial_ids": [],
        }
    inventory = list(results_dir.iterdir())
    nonfiles = sorted(path.name for path in inventory if not path.is_file())
    if nonfiles:
        raise ValueError(f"non-file raw result inventory entries: {nonfiles}")
    results = raw_manifest["results"]
    referenced_names = {entry["path"] for entry in results.values()}
    orphan_paths = sorted(
        (path for path in inventory if path.name not in referenced_names),
        key=lambda path: path.name,
    )
    missing_ids = set(expected_rows) - set(results)
    filename_to_trial = {
        result_filename(trial_id): trial_id for trial_id in missing_ids
    }
    recovered: list[tuple[str, dict[str, str]]] = []
    for path in orphan_paths:
        trial_id = filename_to_trial.get(path.name)
        if trial_id is None:
            raise ValueError(f"unrecognized raw result orphan: {path.name}")
        row = expected_rows[trial_id]
        fixture = _fixture(stack, row)
        expected = _common(stack, fixture, row)
        digest = file_sha256(path)
        entry = {
            "path": path.name,
            "planned_trial_id": trial_id,
            "sha256": digest,
        }
        try:
            payload = _validate_existing_result(
                path, entry=entry, expected=expected
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"raw result orphan failed strict validation: {path.name}"
            ) from error
        if payload.get("planned_trial_id") != trial_id:
            raise ValueError("raw result orphan trial identity mismatch")
        if path.read_bytes() != canonical_json_bytes(payload):
            raise ValueError(
                f"raw result orphan is not canonical JSON: {path.name}"
            )
        recovered.append((trial_id, entry))
    if recovered:
        for trial_id, entry in recovered:
            results[trial_id] = entry
        atomic_write_bytes(
            raw_manifest_path,
            canonical_json_bytes(raw_manifest),
            replace=raw_manifest_path.exists(),
        )
    return {
        "recovered_orphan_result_count": len(recovered),
        "recovered_orphan_trial_ids": [trial_id for trial_id, _ in recovered],
    }


def _validate_existing_result(
    path: Path,
    *,
    entry: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if expected["condition"] == "IDEAL_MATCHED":
        from .phase_a_trial_resume import validate_existing_trial_result_for_resume

        return validate_existing_trial_result_for_resume(
            path, manifest_entry=entry, expected=expected
        )
    from .full_synthetic_trial_result import (
        validate_existing_full_synthetic_trial_result_for_resume,
    )

    return validate_existing_full_synthetic_trial_result_for_resume(
        path, manifest_entry=entry, expected=expected
    )


def _audit_raw_inventory(
    stack: Mapping[str, Any],
    results_dir: Path,
    raw_manifest: Mapping[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
    *,
    allow_unreferenced_files: bool = False,
) -> dict[str, Any]:
    from .phase_a_trial_result_writer import result_filename

    results = raw_manifest["results"]
    manifest_ids = set(results)
    expected_ids = set(expected_rows)
    extra_ids = manifest_ids - expected_ids
    missing_ids = expected_ids - manifest_ids
    corrupt = 0
    checksum_mismatch = 0
    paths: list[str] = []
    payloads: list[dict[str, Any]] = []
    for trial_id in sorted(manifest_ids & expected_ids):
        entry = results[trial_id]
        try:
            if type(entry) is not dict or set(entry) != {
                "path",
                "planned_trial_id",
                "sha256",
            }:
                raise ValueError("raw result entry schema changed")
            path_name = entry["path"]
            if (
                type(path_name) is not str
                or Path(path_name).name != path_name
                or path_name != result_filename(trial_id)
                or entry["planned_trial_id"] != trial_id
            ):
                raise ValueError("raw result entry identity changed")
            paths.append(path_name)
            result_path = results_dir / path_name
            if file_sha256(result_path) != entry["sha256"]:
                checksum_mismatch += 1
                raise ValueError("raw result SHA mismatch")
            row = expected_rows[trial_id]
            fixture = _fixture(stack, row)
            payloads.append(
                _validate_existing_result(
                    result_path,
                    entry=entry,
                    expected=_common(stack, fixture, row),
                )
            )
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            corrupt += 1
    duplicate_paths = len(paths) - len(set(paths))
    inventory = list(results_dir.iterdir()) if results_dir.exists() else []
    actual_files = {path.name for path in inventory if path.is_file()}
    invalid_entries = sum(not path.is_file() for path in inventory)
    manifest_files = {
        entry.get("path")
        for entry in results.values()
        if type(entry) is dict and type(entry.get("path")) is str
    }
    unreferenced_files = actual_files - manifest_files
    reverse_mismatch = (
        len(manifest_files - actual_files)
        + (0 if allow_unreferenced_files else len(unreferenced_files))
        + invalid_entries
    )
    report = {
        "checksum_mismatch_count": checksum_mismatch,
        "corrupt_trial_count": corrupt,
        "duplicate_trial_count": duplicate_paths,
        "extra_trial_count": len(extra_ids),
        "missing_trial_count": len(missing_ids),
        "payloads": payloads,
        "reverse_inventory_mismatch_count": reverse_mismatch,
        "unreferenced_result_files": sorted(unreferenced_files),
    }
    if any(
        report[name] != 0
        for name in (
            "checksum_mismatch_count",
            "corrupt_trial_count",
            "duplicate_trial_count",
            "extra_trial_count",
            "reverse_inventory_mismatch_count",
        )
    ):
        raise ValueError("Confirmatory raw result inventory is corrupt or ambiguous")
    return report


def _completed_raw_manifest_sha256(
    raw_manifest_path: Path, raw_manifest: dict[str, Any]
) -> str:
    """Bind the final canonical raw-result inventory before publication."""

    from .phase_a_trial_result_schema import canonical_json_bytes

    if not raw_manifest_path.is_file():
        raise FileNotFoundError(
            "completed Confirmatory raw result manifest is missing"
        )
    if raw_manifest_path.read_bytes() != canonical_json_bytes(raw_manifest):
        raise ValueError(
            "completed Confirmatory raw result manifest is not the canonical "
            "in-memory inventory"
        )
    return file_sha256(raw_manifest_path)


def execute_synthetic_confirmatory(
    *,
    manifest_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    """Run the exact authorized 595/1,190 matrix with strict resume semantics."""

    if not resume:
        raise PermissionError("Confirmatory formal execution requires explicit --resume")
    from .full_synthetic_development_protocol import assert_isolated_python_runtime

    assert_isolated_python_runtime()
    stack = load_synthetic_confirmatory_stack(
        manifest_path, require_authorized=True
    )
    output = _assert_invocation(
        stack, run_id=run_id, output_dir=output_dir, workers=workers
    )
    from .asset_verifier import source_runtime_import_paths
    from .full_synthetic_snapshot_builder import FullSyntheticSourceAccessMonitor
    from .synthetic_confirmatory_snapshot_builder import (
        prepare_synthetic_confirmatory_snapshots,
    )

    monitor = FullSyntheticSourceAccessMonitor()
    monitor.install()
    preparation = prepare_synthetic_confirmatory_snapshots(
        stack["root"],
        stack["cache_root"],
        stack["snapshots"],
        lock_path=stack["lock_path"],
        resume=True,
    )
    lock = preparation["lock"]
    stack["lock"] = lock
    stack["lock_by_id"] = {
        entry["snapshot_id"]: entry for entry in lock["snapshots"]
    }
    if len(stack["lock_by_id"]) != SNAPSHOT_COUNT:
        raise ValueError("Confirmatory snapshot lock contains duplicate IDs")
    stack["snapshot_lock_sha256"] = file_sha256(stack["lock_path"])

    output.mkdir(parents=True, exist_ok=True)
    results_dir = output / "raw_results"
    raw_manifest_path = output / "raw_result_manifest.json"
    events_path = output / "attempt_events.ndjson"
    raw_manifest = _read_raw_manifest(raw_manifest_path, run_id)
    rows_by_id = {row["planned_trial_id"]: row for row in stack["trials"]}
    if len(rows_by_id) != TRIAL_COUNT:
        raise ValueError("Confirmatory trial plan contains duplicate IDs")
    orphan_recovery = _recover_atomic_result_orphans(
        stack,
        results_dir,
        raw_manifest_path,
        raw_manifest,
        rows_by_id,
    )
    initial = _audit_raw_inventory(stack, results_dir, raw_manifest, rows_by_id)
    pending = [
        row
        for row in stack["trials"]
        if row["planned_trial_id"] not in raw_manifest["results"]
    ]
    from .phase_a_attempt_events import append_attempt_event, read_attempt_events
    from .phase_a_trial_result_schema import canonical_json_bytes
    from .phase_a_trial_result_writer import (
        atomic_write_bytes,
        result_filename,
    )

    for row in pending:
        append_attempt_event(
            events_path,
            planned_trial_id=row["planned_trial_id"],
            snapshot_id=row["planned_snapshot_id"],
            backend=row["backend"],
            event_type="STARTED",
            detail=None,
        )
    completed_this_invocation = 0
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="synthetic-confirmatory"
    ) as executor:
        future_rows = {
            executor.submit(_execute_one, stack, row): row for row in pending
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            common, payload = future.result()
            destination = results_dir / result_filename(row["planned_trial_id"])
            atomic_write_bytes(
                destination, canonical_json_bytes(payload), replace=False
            )
            digest = file_sha256(destination)
            raw_manifest["results"][row["planned_trial_id"]] = {
                "path": destination.name,
                "planned_trial_id": row["planned_trial_id"],
                "sha256": digest,
            }
            atomic_write_bytes(
                raw_manifest_path,
                canonical_json_bytes(raw_manifest),
                replace=True,
            )
            append_attempt_event(
                events_path,
                planned_trial_id=row["planned_trial_id"],
                snapshot_id=row["planned_snapshot_id"],
                backend=common["backend"],
                event_type="COMPLETED",
                detail=None,
            )
            completed_this_invocation += 1
    final = _audit_raw_inventory(stack, results_dir, raw_manifest, rows_by_id)
    payloads = final["payloads"]
    if len(payloads) != TRIAL_COUNT or final["missing_trial_count"] != 0:
        raise RuntimeError("Confirmatory formal matrix is incomplete")
    raw_result_manifest_sha256 = _completed_raw_manifest_sha256(
        raw_manifest_path, raw_manifest
    )
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        by_snapshot[payload["snapshot_id"]].append(payload)
    checksum_fields = (
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
        "snapshot_checksum",
    )
    pairing_mismatches = sum(
        len(rows) != 2
        or {row["backend"] for row in rows} != set(BACKENDS)
        or any(rows[0][name] != rows[1][name] for name in checksum_fields)
        for rows in by_snapshot.values()
    )
    if pairing_mismatches:
        raise ValueError("Confirmatory backend input pairing failed")
    imports = source_runtime_import_paths()
    if monitor.count or imports:
        raise PermissionError("source repository runtime isolation failed")
    backend_counts = Counter(row["backend"] for row in payloads)
    condition_counts = Counter(row["condition"] for row in payloads)
    events = read_attempt_events(events_path)
    event_mismatch = sum(
        event["planned_trial_id"] not in rows_by_id
        or event["snapshot_id"]
        != rows_by_id.get(event["planned_trial_id"], {}).get(
            "planned_snapshot_id"
        )
        or event["backend"]
        != rows_by_id.get(event["planned_trial_id"], {}).get("backend")
        for event in events
    )
    if event_mismatch:
        raise ValueError("Confirmatory attempt-event identity mismatch")
    run_manifest = {
        "backend_execution_count_this_invocation": completed_this_invocation,
        "backend_input_checksum_mismatch_count": pairing_mismatches,
        "completed_snapshot_count": len(by_snapshot),
        "completed_trial_count": len(payloads),
        "condition_trial_counts": dict(sorted(condition_counts.items())),
        "confirmatory_rng_instantiation_count_this_invocation": preparation[
            "confirmatory_rng_instantiation_count_this_invocation"
        ],
        "corrupt_trial_count": final["corrupt_trial_count"],
        "duplicate_trial_count": final["duplicate_trial_count"],
        "event_identity_mismatch_count": event_mismatch,
        "extra_trial_count": final["extra_trial_count"],
        "generated_snapshot_count_this_invocation": preparation[
            "generated_snapshot_count"
        ],
        "missing_trial_count": final["missing_trial_count"],
        "native_execution_count": 0,
        "native_trial_count": 0,
        "nonfinite_output_count": sum(not row["finite_output"] for row in payloads),
        "open3d_trial_count": backend_counts["open3d_point_to_plane"],
        "pcl_trial_count": backend_counts["pcl_point_to_plane"],
        "recovered_orphan_result_count": orphan_recovery[
            "recovered_orphan_result_count"
        ],
        "raw_result_manifest_sha256": raw_result_manifest_sha256,
        "resume_skipped_valid_result_count": len(initial["payloads"]),
        "resumed_snapshot_count_this_invocation": preparation[
            "resumed_snapshot_count"
        ],
        "reverse_raw_result_inventory_mismatch_count": final[
            "reverse_inventory_mismatch_count"
        ],
        "run_id": run_id,
        "schema_version": FORMAL_RUN_SCHEMA,
        "scientific_solver_failure_count": sum(row["solver_failure"] for row in payloads),
        "snapshot_lock_sha256": stack["snapshot_lock_sha256"],
        "source_repository_runtime_file_read_count": monitor.count,
        "source_repository_runtime_import_count": len(imports),
        "source_repository_runtime_import_paths": imports,
        "trial_result_checksum_mismatch_count": final[
            "checksum_mismatch_count"
        ],
        "workers": workers,
    }
    atomic_write_bytes(
        output / "run_manifest.json",
        canonical_json_bytes(run_manifest),
        replace=True,
    )
    return run_manifest


__all__ = [
    "DRY_RUN_SCHEMA",
    "FORMAL_RUN_SCHEMA",
    "dry_run_synthetic_confirmatory",
    "execute_synthetic_confirmatory",
    "load_synthetic_confirmatory_stack",
]
