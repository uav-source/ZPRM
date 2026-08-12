"""Fail-closed authorization for Boreas v2 Stage-2 payload preparation.

The JSON artifact is an auditable declaration, not a signature.  A caller may
only obtain :class:`VerifiedStage2Authorization` after this module has
recomputed every live authority under the exact clean Git commit recorded in
the artifact.  Production download and execution objects require that in-memory
capability and the same active :class:`NoRegistrationGuard`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .boreas_external_v2_stage1_verifier import (
    BoreasExternalV2Stage1VerificationError,
    verify_boreas_external_v2_stage1,
)
from .boreas_stage2_storage_optimization_verifier import (
    BoreasStage2StorageVerificationError,
    verify_boreas_v2_stage2_storage_plan,
)
from .guard import (
    NoRegistrationGuard,
    RegistrationForbiddenError,
    assert_preparation_sources_are_safe,
)
from .io import (
    PreparationIOError,
    atomic_write_bytes,
    canonical_json_bytes,
    compact_sha256,
    sha256_file,
)
from .stage2_disk_gate import Stage2DiskGate


AUTHORIZATION_SCHEMA = "boreas_v2_stage2_download_authorization_v2"
EXPECTED_BRANCH = "run/boreas-v2-stage2-data-preparation"
EXPECTED_STAGE1_MANIFEST_FILE_SHA256 = (
    "71cfd78aa588c67dd28f8f8be87b7514c20ed090e75a8433583362253a97a8be"
)
EXPECTED_STAGE1_MANIFEST_ROOT_SHA256 = (
    "677346ded7d4d45be1877dc19d0c316e80e87699784a441ed1d8b69848eae44f"
)
EXPECTED_STORAGE_MANIFEST_FILE_SHA256 = (
    "51ac1448f30461ceb8c0dc93c8c33e084841cd088c8321698a1cdd338cfa4a9c"
)
EXPECTED_STORAGE_MANIFEST_ROOT_SHA256 = (
    "c00aa59eef3f602316cf2c25e1c26fd0573b9d63dcdd50b3c766463291d1fe73"
)
EXPECTED_PAIR_SHA256 = (
    "b28a52498a2fddc1b80d43c626b75179358066677ce8f493be22afba7a19639c"
)
EXPECTED_ALLOWLIST_SHA256 = (
    "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
)
EXPECTED_STORAGE_CONTRACT_SHA256 = (
    "e2cc29ba56e2ea9a0cdadb076c908ee59777b45e8a7b6dc560083978683cd3aa"
)
EXPECTED_STORAGE_BUDGET_SHA256 = (
    "d2d0f5748bb9531f5727284b6449e56c11789954cf1127423727b64d62b7d13f"
)
EXPECTED_MAP_SEQUENCE = "boreas-2021-11-14-09-47"
EXPECTED_QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
EXPECTED_BUCKET = "boreas"
EXPECTED_OBJECT_COUNT = 20_061
EXPECTED_REMOTE_BYTES = 104_158_637_472
MINIMUM_START_FREE_BYTES = 101_563_921_871
RUNTIME_LOW_DISK_WATERMARK_BYTES = 16_927_320_312
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SELF_HASH_SEMANTICS = "UNKEYED_SHA256_INTEGRITY_ONLY_LIVE_REAUTHENTICATION_REQUIRED"
ROOT_NAMES = (
    "repository",
    "stage1_data",
    "stage2_runtime",
    "stage2_temporary",
    "monitored_disk",
)
AUTHORIZATION_UNSIGNED_FIELDS = frozenset(
    {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
        "PUBLIC_DATA_V2_RUN_AUTHORIZED",
        "REAL_REGISTRATION_AUTHORIZED",
        "STAGE2_DOWNLOAD_AUTHORIZED",
        "actual_trials",
        "allowlist_object_count",
        "allowlist_remote_bytes",
        "allowlist_sha256",
        "authorization_path",
        "authorization_scope",
        "branch",
        "bucket",
        "commit",
        "disk_free_bytes_at_authorization",
        "execution_mode",
        "extrinsic_limitation",
        "minimum_start_free_disk_bytes",
        "no_icp_guard_active",
        "no_registration_environment_active",
        "preprocessing_contract_payload_sha256",
        "preprocessing_contract_sha256",
        "primary_pair",
        "primary_pair_sha256",
        "registration_execution_count",
        "root_bindings",
        "runtime_low_disk_watermark_bytes",
        "schema_version",
        "self_hash_semantics",
        "stage1_manifest_file_sha256",
        "stage1_verification_report_path",
        "stage1_verification_report_sha256",
        "static_source_audit",
        "storage_budget_sha256",
        "storage_contract_sha256",
        "storage_manifest_file_sha256",
        "storage_verification",
        "timestamp_utc",
        "worktree_clean_before_authorization",
    }
)


class BoreasStage2AuthorizationError(RuntimeError):
    """The narrow Stage-2 preparation authority could not be established."""


def _git(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository, text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise BoreasStage2AuthorizationError(
            f"git {' '.join(arguments)} failed: {exc.stderr.strip()}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BoreasStage2AuthorizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, require_canonical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise BoreasStage2AuthorizationError(f"required JSON is absent or unsafe: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except BoreasStage2AuthorizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoreasStage2AuthorizationError(f"cannot parse required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BoreasStage2AuthorizationError(f"required JSON is not an object: {path}")
    if require_canonical and payload != canonical_json_bytes(value):
        raise BoreasStage2AuthorizationError(f"required JSON is not canonical: {path}")
    return value


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _same(actual: Any, expected: Any, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise BoreasStage2AuthorizationError(
            f"{label} differs: expected {expected!r}, got {actual!r}"
        )


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BoreasStage2AuthorizationError(f"{label} must be a nonnegative integer")
    return value


def _timestamp(now: Callable[[], datetime] | None) -> str:
    value = (now or (lambda: datetime.now(timezone.utc)))()
    if value.tzinfo is None or value.utcoffset() is None:
        raise BoreasStage2AuthorizationError("authorization clock must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _validate_recorded_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise BoreasStage2AuthorizationError("authorization timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BoreasStage2AuthorizationError("authorization timestamp is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
        or not value.endswith("Z")
    ):
        raise BoreasStage2AuthorizationError("authorization timestamp must be canonical UTC")
    return value


def _canonical_directory(value: str | Path, *, label: str, writable: bool) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise BoreasStage2AuthorizationError(
            f"{label} must be an existing absolute non-symlink directory"
        )
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise BoreasStage2AuthorizationError(f"{label} must be a canonical path")
    access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not os.access(candidate, access):
        raise BoreasStage2AuthorizationError(f"{label} lacks required filesystem access")
    return candidate


def _within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _root_bindings(
    *,
    repository: str | Path,
    data_root: str | Path,
    runtime_root: str | Path,
    temporary_root: str | Path,
    monitored_disk_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    roots = {
        "repository": _canonical_directory(
            repository, label="repository", writable=False
        ),
        "stage1_data": _canonical_directory(
            data_root, label="Stage-1 data root", writable=False
        ),
        "stage2_runtime": _canonical_directory(
            runtime_root, label="Stage-2 runtime root", writable=True
        ),
        "stage2_temporary": _canonical_directory(
            temporary_root, label="Stage-2 temporary root", writable=True
        ),
        "monitored_disk": _canonical_directory(
            monitored_disk_path, label="monitored disk path", writable=True
        ),
    }
    repository_path = roots["repository"]
    data_path = roots["stage1_data"]
    runtime_path = roots["stage2_runtime"]
    temporary_path = roots["stage2_temporary"]
    if _within(repository_path, runtime_path) or _within(runtime_path, repository_path):
        raise BoreasStage2AuthorizationError(
            "Stage-2 runtime root must not overlap the repository"
        )
    if _within(data_path, runtime_path) or _within(runtime_path, data_path):
        raise BoreasStage2AuthorizationError(
            "Stage-2 runtime root must not overlap frozen Stage-1 data"
        )
    if temporary_path == runtime_path or not _within(runtime_path, temporary_path):
        raise BoreasStage2AuthorizationError(
            "Stage-2 temporary root must be a strict child of the runtime root"
        )
    devices = {name: int(path.stat().st_dev) for name, path in roots.items()}
    write_devices = {
        devices["stage2_runtime"],
        devices["stage2_temporary"],
        devices["monitored_disk"],
    }
    if len(write_devices) != 1:
        raise BoreasStage2AuthorizationError(
            "runtime, temporary, and monitored paths must be on the same filesystem"
        )
    bindings = {
        name: {
            "access": "READ_WRITE_EXECUTE" if name in {
                "stage2_runtime",
                "stage2_temporary",
                "monitored_disk",
            } else "READ_EXECUTE",
            "path": str(path),
            "st_dev": devices[name],
        }
        for name, path in roots.items()
    }
    return bindings, roots


def _canonical_output_path(output_path: str | Path, *, runtime_root: Path) -> Path:
    output = Path(output_path)
    if not output.is_absolute():
        raise BoreasStage2AuthorizationError("authorization output path must be absolute")
    if output.is_symlink():
        raise BoreasStage2AuthorizationError("authorization output cannot be a symlink")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise BoreasStage2AuthorizationError("authorization output parent is unsafe")
    if output.parent / output.name != output or not _within(runtime_root, output):
        raise BoreasStage2AuthorizationError(
            "authorization output must be a canonical path inside the Stage-2 runtime root"
        )
    return output


def _assert_active_guard(no_registration_guard: NoRegistrationGuard) -> None:
    if not isinstance(no_registration_guard, NoRegistrationGuard):
        raise BoreasStage2AuthorizationError(
            "authorization requires a NoRegistrationGuard instance"
        )
    if not no_registration_guard.active:
        raise BoreasStage2AuthorizationError("NoRegistrationGuard is not active")
    patched_names = {"Popen", "run", "call", "check_call", "check_output"}
    if set(no_registration_guard._original_subprocess) != patched_names:
        raise BoreasStage2AuthorizationError(
            "NoRegistrationGuard does not own the active subprocess patch set"
        )
    for name in patched_names:
        wrapper = getattr(subprocess, name)
        closure = getattr(wrapper, "__closure__", None) or ()
        if not any(cell.cell_contents is no_registration_guard for cell in closure):
            raise BoreasStage2AuthorizationError(
                "NoRegistrationGuard object identity differs from the active patches"
            )
    if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise BoreasStage2AuthorizationError(
            "ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is required"
        )


def _git_state(repository: Path) -> dict[str, str]:
    branch = _git(repository, "branch", "--show-current")
    head = _git(repository, "rev-parse", "HEAD")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != EXPECTED_BRANCH:
        raise BoreasStage2AuthorizationError(
            f"authorization requires branch {EXPECTED_BRANCH}"
        )
    if COMMIT_RE.fullmatch(head) is None:
        raise BoreasStage2AuthorizationError("authorization HEAD is not a full commit SHA")
    if status:
        raise BoreasStage2AuthorizationError("authorization requires a clean worktree")
    return {"branch": branch, "commit": head, "status": status}


def _closure_candidate_inventory(
    repository: Path, runtime_root: Path
) -> tuple[Path, dict[str, dict[str, Any]]]:
    from .boreas_v2_stage2_preparation_verifier import REQUIRED_FILES

    candidate = runtime_root / "checkpoints/final_closure_candidate"
    if (
        candidate.is_symlink()
        or not candidate.is_dir()
        or candidate.resolve(strict=True) != candidate
    ):
        raise BoreasStage2AuthorizationError(
            "publication worktree exception lacks the fixed closure candidate"
        )
    manifest_path = candidate / "boreas_v2_stage2_frozen_manifest.json"
    manifest = _load_json(manifest_path, require_canonical=True)
    unsigned = dict(manifest)
    claim = unsigned.pop("manifest_root_sha256", None)
    if claim != compact_sha256(unsigned):
        raise BoreasStage2AuthorizationError(
            "publication closure candidate manifest self-hash differs"
        )
    payload = manifest.get("payload")
    if not isinstance(payload, list):
        raise BoreasStage2AuthorizationError(
            "publication closure candidate payload inventory is absent"
        )
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size_bytes"}
            or not isinstance(row["path"], str)
            or Path(row["path"]).name != row["path"]
            or row["path"] in rows
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or not isinstance(row["sha256"], str)
            or SHA256_RE.fullmatch(row["sha256"]) is None
        ):
            raise BoreasStage2AuthorizationError(
                "publication closure candidate payload row differs"
            )
        rows[row["path"]] = dict(row)
    required = set(rows) | {
        "boreas_v2_stage2_frozen_manifest.json",
        "SHA256SUMS",
    }
    if required != set(REQUIRED_FILES):
        raise BoreasStage2AuthorizationError(
            "publication closure candidate required-file inventory differs"
        )
    if {entry.name for entry in candidate.iterdir()} != required:
        raise BoreasStage2AuthorizationError(
            "publication closure candidate inventory is open"
        )
    for name in required:
        path = candidate / name
        metadata = os.lstat(path)
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or path.resolve(strict=True) != path
        ):
            raise BoreasStage2AuthorizationError(
                "publication closure candidate contains an unsafe file"
            )
    for name, row in rows.items():
        path = candidate / name
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise BoreasStage2AuthorizationError(
                "publication closure candidate bytes differ from its manifest"
            )
    return candidate, {
        name: {
            "sha256": sha256_file(candidate / name),
            "size_bytes": (candidate / name).stat().st_size,
        }
        for name in required
    }


def _git_state_allowing_exact_closure_publication(
    repository: Path, runtime_root: Path
) -> dict[str, str]:
    """Narrow resume exception for one candidate-identical untracked closure."""

    branch = _git(repository, "branch", "--show-current")
    head = _git(repository, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or COMMIT_RE.fullmatch(head) is None:
        raise BoreasStage2AuthorizationError(
            "publication resume Git branch/HEAD differs"
        )
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if not status:
        return {"branch": branch, "commit": head, "status": ""}
    candidate, inventory = _closure_candidate_inventory(repository, runtime_root)
    del candidate
    destination = repository / "frozen_assets/boreas_v2_stage2_preparation"
    intent_path = runtime_root / "checkpoints/closure_publication_intent.json"
    permitted_roots = {destination}
    if intent_path.exists() or intent_path.is_symlink():
        intent = _load_json(intent_path, require_canonical=True)
        unsigned_intent = dict(intent)
        intent_claim = unsigned_intent.pop("intent_sha256", None)
        if (
            intent_claim != compact_sha256(unsigned_intent)
            or intent.get("schema")
            != "zprm.boreas.v2.stage2.closure_publication_intent.v1"
            or intent.get("runtime_root") != str(runtime_root)
            or intent.get("destination_root") != str(destination)
            or intent.get("candidate_root")
            != str(runtime_root / "checkpoints/final_closure_candidate")
            or not isinstance(intent.get("staging_root"), str)
        ):
            raise BoreasStage2AuthorizationError(
                "publication resume intent differs"
            )
        staging = Path(intent["staging_root"])
        if (
            staging.parent != destination.parent
            or staging.name
            != f".{destination.name}.{intent.get('manifest_root_sha256')}.staging"
        ):
            raise BoreasStage2AuthorizationError(
                "publication resume staging binding differs"
            )
        permitted_roots.add(staging)
    observed_roots: set[Path] = set()
    for line in status.splitlines():
        if not line.startswith("?? "):
            raise BoreasStage2AuthorizationError(
                "publication worktree exception found a tracked source change"
            )
        relative = Path(line[3:])
        absolute = repository / relative
        matches = [
            root
            for root in permitted_roots
            if absolute == root or root in absolute.parents
        ]
        if len(matches) != 1:
            raise BoreasStage2AuthorizationError(
                "publication worktree exception found an unrelated file"
            )
        observed_roots.add(matches[0])
    if len(observed_roots) != 1:
        raise BoreasStage2AuthorizationError(
            "publication worktree exception spans multiple roots"
        )
    published = next(iter(observed_roots))
    if (
        published.is_symlink()
        or not published.is_dir()
        or published.resolve(strict=True) != published
        or {entry.name for entry in published.iterdir()} != set(inventory)
    ):
        raise BoreasStage2AuthorizationError(
            "publication worktree exception inventory differs"
        )
    for name, identity in inventory.items():
        path = published / name
        metadata = os.lstat(path)
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or path.resolve(strict=True) != path
            or metadata.st_size != identity["size_bytes"]
            or sha256_file(path) != identity["sha256"]
        ):
            raise BoreasStage2AuthorizationError(
                "publication worktree exception bytes differ from candidate"
            )
    return {"branch": branch, "commit": head, "status": ""}


def _authenticate_preprocessing(repository: Path) -> tuple[dict[str, Any], str]:
    path = repository / "protocols/boreas_v2_stage2_preprocessing_contract.json"
    contract = _load_json(path, require_canonical=True)
    _same(contract.get("contract_status"), "FROZEN", "preprocessing contract status")
    supplied = contract.get("contract_payload_sha256")
    if not isinstance(supplied, str) or SHA256_RE.fullmatch(supplied) is None:
        raise BoreasStage2AuthorizationError("preprocessing self-hash is invalid")
    unsigned = {
        key: value
        for key, value in contract.items()
        if key != "contract_payload_sha256"
    }
    _same(supplied, compact_sha256(unsigned), "preprocessing canonical self-hash")
    bindings = contract.get("implementation_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise BoreasStage2AuthorizationError(
            "preprocessing implementation bindings are absent"
        )
    expected_binding_fields = {
        "production_preprocessing": {"path", "file_sha256"},
        "independent_canonical_source_witness": {
            "path",
            "file_sha256",
            "verification_claim",
        },
    }
    if set(bindings) != set(expected_binding_fields):
        raise BoreasStage2AuthorizationError(
            "preprocessing implementation binding names differ"
        )
    for name, binding in bindings.items():
        if (
            not isinstance(binding, dict)
            or set(binding) != expected_binding_fields[name]
        ):
            raise BoreasStage2AuthorizationError(
                f"preprocessing implementation binding is malformed: {name}"
            )
        if name == "independent_canonical_source_witness":
            _same(
                binding["verification_claim"],
                "DUAL_PATH_BYTE_IDENTITY_USING_SHARED_FROZEN_PREPROCESSING_PRIMITIVES",
                "canonical-source witness verification claim",
            )
        relative = Path(binding["path"])
        implementation = repository / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or implementation.is_symlink()
            or not implementation.is_file()
            or implementation.resolve(strict=True) != implementation
        ):
            raise BoreasStage2AuthorizationError(
                f"preprocessing implementation is absent or unsafe: {name}"
            )
        expected = binding["file_sha256"]
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            raise BoreasStage2AuthorizationError(
                f"preprocessing implementation SHA is invalid: {name}"
            )
        _same(
            sha256_file(implementation),
            expected,
            f"preprocessing implementation SHA-256 {name}",
        )
    return contract, sha256_file(path)


def _authenticate_frozen_inputs(repository: Path) -> dict[str, str]:
    stage1 = repository / "frozen_assets/public_data_external_validation_v2_boreas_stage1"
    storage = repository / "frozen_assets/boreas_v2_stage2_storage_optimization"
    paths = {
        "stage1_manifest": stage1 / "frozen_manifest.json",
        "storage_manifest": storage / "frozen_manifest.json",
        "primary_pair": stage1 / "boreas_v2_pair_selection.json",
        "allowlist": stage1 / "boreas_v2_stage2_download_allowlist.csv",
        "storage_contract": storage / "stage2_storage_architecture_contract.json",
        "storage_budget": storage / "boreas_v2_stage2_disk_budget_optimized.json",
    }
    expected = {
        "stage1_manifest": EXPECTED_STAGE1_MANIFEST_FILE_SHA256,
        "storage_manifest": EXPECTED_STORAGE_MANIFEST_FILE_SHA256,
        "primary_pair": EXPECTED_PAIR_SHA256,
        "allowlist": EXPECTED_ALLOWLIST_SHA256,
        "storage_contract": EXPECTED_STORAGE_CONTRACT_SHA256,
        "storage_budget": EXPECTED_STORAGE_BUDGET_SHA256,
    }
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise BoreasStage2AuthorizationError(f"frozen {name} is absent or unsafe")
        _same(sha256_file(path), expected[name], f"frozen {name} SHA-256")
    _same(
        _load_json(paths["stage1_manifest"]).get("manifest_root_sha256"),
        EXPECTED_STAGE1_MANIFEST_ROOT_SHA256,
        "Stage-1 manifest root",
    )
    _same(
        _load_json(paths["storage_manifest"]).get("manifest_root_sha256"),
        EXPECTED_STORAGE_MANIFEST_ROOT_SHA256,
        "storage manifest root",
    )
    return expected


def _independently_verify_stage1(
    *, repository: Path, data_root: Path, supplied_report_path: Path | None
) -> tuple[dict[str, Any], str]:
    try:
        report = verify_boreas_external_v2_stage1(
            repository=repository,
            data_root=data_root,
            runtime_root=(
                repository
                / "frozen_assets/public_data_external_validation_v2_boreas_stage1"
            ),
        )
    except (BoreasExternalV2Stage1VerificationError, OSError, ValueError) as exc:
        raise BoreasStage2AuthorizationError(
            f"independent Stage-1 verification failed: {exc}"
        ) from exc
    required = {
        "BOREAS_EXTERNAL_V2_STAGE1_READY": True,
        "BOREAS_EXTERNAL_V2_STAGE1_VERIFICATION_PASS": True,
        "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD": True,
        "manifest_root_sha256": EXPECTED_STAGE1_MANIFEST_ROOT_SHA256,
        "primary_pair": {
            "map_sequence_id": EXPECTED_MAP_SEQUENCE,
            "query_sequence_id": EXPECTED_QUERY_SEQUENCE,
        },
        "registration_execution_count": 0,
        "selected_stage2_lidar_object_count": EXPECTED_OBJECT_COUNT,
        "verification_pass": True,
    }
    for key, expected in required.items():
        _same(report.get(key), expected, f"independent Stage-1 report {key}")
    if supplied_report_path is not None:
        supplied = _load_json(supplied_report_path, require_canonical=True)
        _same(supplied, report, "supplied/independently recomputed Stage-1 report")
    payload = canonical_json_bytes(report)
    return report, hashlib.sha256(payload).hexdigest()


def _stage1_report_evidence_path(runtime_root: Path, *, create: bool) -> Path:
    evidence = runtime_root / "evidence"
    if create and not evidence.exists():
        evidence.mkdir(mode=0o700)
    if (
        evidence.is_symlink()
        or not evidence.is_dir()
        or evidence.resolve(strict=True) != evidence
    ):
        raise BoreasStage2AuthorizationError(
            "Stage-1 verification evidence directory is unsafe"
        )
    return evidence / "boreas_v2_stage1_verification_live.json"


def _freeze_stage1_report_evidence(
    *, runtime_root: Path, report: Mapping[str, Any]
) -> Path:
    path = _stage1_report_evidence_path(runtime_root, create=True)
    payload = canonical_json_bytes(report)
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.read_bytes() != payload
        ):
            raise BoreasStage2AuthorizationError(
                "existing Stage-1 live verification evidence differs or is unsafe"
            )
    else:
        atomic_write_bytes(path, payload)
    return path


def _independently_verify_storage(
    *, repository: Path, data_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report = verify_boreas_v2_stage2_storage_plan(
            repository=repository,
            runtime_root=(
                repository / "frozen_assets/boreas_v2_stage2_storage_optimization"
            ),
            data_root=data_root,
        )
    except (BoreasStage2StorageVerificationError, OSError, ValueError) as exc:
        raise BoreasStage2AuthorizationError(
            f"independent storage verification failed: {exc}"
        ) from exc
    required = {
        "BOREAS_V2_STAGE2_STORAGE_VERIFICATION_PASS": True,
        "STAGE2_STORAGE_PLAN_VERIFICATION_PASS": True,
        "CURRENT_DISK_SUFFICIENT": True,
        "STAGE2_STORAGE_PLAN_READY": True,
        "allowlist_object_count": EXPECTED_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "local_lidar_bin_count": 0,
        "manifest_root_sha256": EXPECTED_STORAGE_MANIFEST_ROOT_SHA256,
        "real_execution_count": 0,
        "registration_execution_count": 0,
        "lidar_payload_download_count": 0,
        "verification_pass": True,
    }
    for key, expected in required.items():
        _same(report.get(key), expected, f"independent storage report {key}")
    projection = {key: report[key] for key in sorted(required)}
    projection["report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report, projection


def _static_audit(repository: Path) -> dict[str, Any]:
    try:
        result = assert_preparation_sources_are_safe(
            repository / "src/phase_a_harness/real_data_preparation"
        )
    except RegistrationForbiddenError as exc:
        raise BoreasStage2AuthorizationError(f"preparation source audit failed: {exc}") from exc
    if not isinstance(result, dict):
        raise BoreasStage2AuthorizationError("preparation source audit is malformed")
    _same(result.get("pass"), True, "preparation source audit pass")
    return result


def _live_authorities(
    *,
    repository: Path,
    data_root: Path,
    supplied_stage1_report: Path | None,
) -> dict[str, Any]:
    frozen = _authenticate_frozen_inputs(repository)
    preprocessing, preprocessing_file_sha = _authenticate_preprocessing(repository)
    stage1_report, stage1_report_sha = _independently_verify_stage1(
        repository=repository,
        data_root=data_root,
        supplied_report_path=supplied_stage1_report,
    )
    _, storage_projection = _independently_verify_storage(
        repository=repository, data_root=data_root
    )
    return {
        "frozen": frozen,
        "preprocessing_contract_payload_sha256": preprocessing[
            "contract_payload_sha256"
        ],
        "preprocessing_contract_sha256": preprocessing_file_sha,
        "stage1_verification_report": stage1_report,
        "stage1_verification_report_sha256": stage1_report_sha,
        "static_source_audit": _static_audit(repository),
        "storage_verification": storage_projection,
    }


def _authorization_unsigned(
    *,
    git_state: Mapping[str, str],
    roots: Mapping[str, Mapping[str, Any]],
    output: Path,
    live: Mapping[str, Any],
    stage1_report_path: Path,
    disk_free_bytes: int,
    timestamp_utc: str,
) -> dict[str, Any]:
    frozen = live["frozen"]
    value: dict[str, Any] = {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "STAGE2_DOWNLOAD_AUTHORIZED": True,
        "actual_trials": 0,
        "allowlist_object_count": EXPECTED_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "allowlist_sha256": frozen["allowlist"],
        "authorization_path": str(output),
        "authorization_scope": "BOREAS_V2_STAGE2_LIDAR_DATA_PREPARATION_ONLY",
        "branch": git_state["branch"],
        "bucket": EXPECTED_BUCKET,
        "commit": git_state["commit"],
        "disk_free_bytes_at_authorization": disk_free_bytes,
        "execution_mode": "STREAMING_LOW_DISK",
        "extrinsic_limitation": "PASS_WITH_DOCUMENTED_LIMITATION",
        "minimum_start_free_disk_bytes": MINIMUM_START_FREE_BYTES,
        "no_icp_guard_active": True,
        "no_registration_environment_active": True,
        "preprocessing_contract_payload_sha256": live[
            "preprocessing_contract_payload_sha256"
        ],
        "preprocessing_contract_sha256": live["preprocessing_contract_sha256"],
        "primary_pair": {
            "map_sequence_id": EXPECTED_MAP_SEQUENCE,
            "query_sequence_id": EXPECTED_QUERY_SEQUENCE,
        },
        "primary_pair_sha256": frozen["primary_pair"],
        "registration_execution_count": 0,
        "root_bindings": dict(roots),
        "runtime_low_disk_watermark_bytes": RUNTIME_LOW_DISK_WATERMARK_BYTES,
        "schema_version": AUTHORIZATION_SCHEMA,
        "self_hash_semantics": SELF_HASH_SEMANTICS,
        "stage1_manifest_file_sha256": frozen["stage1_manifest"],
        "stage1_verification_report_path": str(stage1_report_path),
        "stage1_verification_report_sha256": live[
            "stage1_verification_report_sha256"
        ],
        "static_source_audit": live["static_source_audit"],
        "storage_budget_sha256": frozen["storage_budget"],
        "storage_contract_sha256": frozen["storage_contract"],
        "storage_manifest_file_sha256": frozen["storage_manifest"],
        "storage_verification": live["storage_verification"],
        "timestamp_utc": timestamp_utc,
        "worktree_clean_before_authorization": True,
    }
    _same(set(value), set(AUTHORIZATION_UNSIGNED_FIELDS), "authorization field set")
    return value


_CAPABILITY_TOKEN = object()
_FORMAL_VERIFICATION_TOKEN = object()


@dataclass(frozen=True)
class VerifiedStage2Authorization:
    """Live-recomputed, process-local capability required by payload code."""

    _token: object = field(repr=False, compare=False)
    _document_bytes: bytes = field(repr=False, compare=False)
    repository: Path
    stage1_data_root: Path
    runtime_root: Path
    temporary_root: Path
    monitored_disk_path: Path
    authorization_path: Path
    authorization_file_sha256: str
    no_registration_guard: NoRegistrationGuard = field(repr=False, compare=False)
    _formal_verification_token: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._token is not _CAPABILITY_TOKEN:
            raise BoreasStage2AuthorizationError(
                "VerifiedStage2Authorization can only be minted by live verification"
            )

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("VerifiedStage2Authorization is process-local and non-serializable")

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self._document_bytes.decode("utf-8"))

    @property
    def formally_verified(self) -> bool:
        """True only for capabilities returned by the full production verifier."""

        return self._formal_verification_token is _FORMAL_VERIFICATION_TOKEN

    @property
    def allowlist_sha256(self) -> str:
        return str(self.document["allowlist_sha256"])

    @property
    def expected_object_count(self) -> int:
        return int(self.document["allowlist_object_count"])

    @property
    def expected_remote_bytes(self) -> int:
        return int(self.document["allowlist_remote_bytes"])

    @property
    def primary_pair(self) -> dict[str, str]:
        return dict(self.document["primary_pair"])

    @property
    def bucket(self) -> str:
        return str(self.document["bucket"])

    def _assert_guard_and_artifact(self) -> None:
        if self._token is not _CAPABILITY_TOKEN:
            raise BoreasStage2AuthorizationError(
                "verified authorization capability token is invalid"
            )
        _assert_active_guard(self.no_registration_guard)
        if (
            self.authorization_path.is_symlink()
            or not self.authorization_path.is_file()
            or self.authorization_path.resolve(strict=True) != self.authorization_path
            or sha256_file(self.authorization_path) != self.authorization_file_sha256
        ):
            raise BoreasStage2AuthorizationError(
                "verified authorization artifact changed or became unsafe"
            )

    def assert_operation_live(self) -> "VerifiedStage2Authorization":
        """Cheap per-object check: active guard, immutable artifact, and devices."""

        self._assert_guard_and_artifact()
        bindings, _ = _root_bindings(
            repository=self.repository,
            data_root=self.stage1_data_root,
            runtime_root=self.runtime_root,
            temporary_root=self.temporary_root,
            monitored_disk_path=self.monitored_disk_path,
        )
        _same(bindings, self.document["root_bindings"], "live root bindings")
        return self

    def assert_live(self) -> "VerifiedStage2Authorization":
        """Full cheap-source gate used before each execution or resume."""

        self.assert_operation_live()
        state = _git_state(self.repository)
        _same(state["branch"], self.document["branch"], "live authorization branch")
        _same(state["commit"], self.document["commit"], "live authorization HEAD")
        return self

    def bind_disk_gate(self, disk_gate: Stage2DiskGate) -> None:
        if not isinstance(disk_gate, Stage2DiskGate):
            raise BoreasStage2AuthorizationError(
                "verified authorization requires a Stage2DiskGate"
            )
        if disk_gate.monitored_path != self.monitored_disk_path:
            raise BoreasStage2AuthorizationError(
                "disk gate monitors a path outside the authorization"
            )
        thresholds = disk_gate.thresholds
        expected = self.document
        _same(
            thresholds.minimum_start_free_bytes,
            expected["minimum_start_free_disk_bytes"],
            "disk gate start threshold",
        )
        _same(
            thresholds.runtime_low_disk_watermark_bytes,
            expected["runtime_low_disk_watermark_bytes"],
            "disk gate runtime watermark",
        )
        _same(
            thresholds.storage_budget_sha256,
            expected["storage_budget_sha256"],
            "disk gate storage budget SHA",
        )
        _same(
            thresholds.storage_mode,
            "RECOMMENDED_OPERATIONAL",
            "disk gate storage mode",
        )
        audit_parent = disk_gate.audit_log_path.parent.resolve(strict=True)
        if not _within(self.runtime_root, audit_parent):
            raise BoreasStage2AuthorizationError(
                "disk gate audit path is outside the authorized runtime root"
            )


def build_boreas_v2_stage2_download_authorization(
    *,
    repository: str | Path,
    data_root: str | Path,
    runtime_root: str | Path,
    temporary_root: str | Path,
    monitored_disk_path: str | Path,
    stage1_verification_report: str | Path | None = None,
    output_path: str | Path,
    no_registration_guard: NoRegistrationGuard,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Write the declaration only after independent live verification passes."""

    _assert_active_guard(no_registration_guard)
    root_bindings, roots = _root_bindings(
        repository=repository,
        data_root=data_root,
        runtime_root=runtime_root,
        temporary_root=temporary_root,
        monitored_disk_path=monitored_disk_path,
    )
    repo = roots["repository"]
    output = _canonical_output_path(output_path, runtime_root=roots["stage2_runtime"])
    if output.exists() or output.is_symlink():
        raise BoreasStage2AuthorizationError(
            "authorization output must not already exist"
        )
    report_path: Path | None = None
    if stage1_verification_report is not None:
        candidate = Path(stage1_verification_report)
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve(strict=True) != candidate
        ):
            raise BoreasStage2AuthorizationError(
                "optional Stage-1 verification report is absent or unsafe"
            )
        report_path = candidate

    initial_git = _git_state(repo)
    live = _live_authorities(
        repository=repo,
        data_root=roots["stage1_data"],
        supplied_stage1_report=report_path,
    )
    free_bytes = _strict_nonnegative_int(
        int(shutil.disk_usage(roots["monitored_disk"]).free), "live free bytes"
    )
    if free_bytes < MINIMUM_START_FREE_BYTES:
        raise BoreasStage2AuthorizationError(
            f"BLOCKED_INSUFFICIENT_DISK: {free_bytes} < {MINIMUM_START_FREE_BYTES}"
        )
    final_git = _git_state(repo)
    _same(final_git, initial_git, "authorization Git state before write")
    _assert_active_guard(no_registration_guard)
    if output.exists() or output.is_symlink():
        raise BoreasStage2AuthorizationError("authorization output appeared during verification")

    stage1_evidence_path = _freeze_stage1_report_evidence(
        runtime_root=roots["stage2_runtime"],
        report=live["stage1_verification_report"],
    )
    unsigned = _authorization_unsigned(
        git_state=initial_git,
        roots=root_bindings,
        output=output,
        live=live,
        stage1_report_path=stage1_evidence_path,
        disk_free_bytes=free_bytes,
        timestamp_utc=_timestamp(now),
    )
    value = dict(unsigned)
    value["authorization_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    atomic_write_bytes(output, canonical_json_bytes(value))
    if _load_json(output, require_canonical=True) != value:
        raise BoreasStage2AuthorizationError("written authorization did not verify")
    return value


def verify_boreas_v2_stage2_download_authorization(
    *,
    repository: str | Path,
    data_root: str | Path,
    runtime_root: str | Path,
    temporary_root: str | Path,
    monitored_disk_path: str | Path,
    authorization_path: str | Path,
    no_registration_guard: NoRegistrationGuard,
    allow_exact_closure_publication_resume: bool = False,
) -> VerifiedStage2Authorization:
    """Recompute all authorities and mint a capability for one run/resume."""

    _assert_active_guard(no_registration_guard)
    root_bindings, roots = _root_bindings(
        repository=repository,
        data_root=data_root,
        runtime_root=runtime_root,
        temporary_root=temporary_root,
        monitored_disk_path=monitored_disk_path,
    )
    path = Path(authorization_path)
    if not path.is_absolute() or path.is_symlink():
        raise BoreasStage2AuthorizationError(
            "authorization path must be absolute and non-symlink"
        )
    path = path.resolve(strict=True)
    expected_path = _canonical_output_path(path, runtime_root=roots["stage2_runtime"])
    value = _load_json(expected_path, require_canonical=True)
    if set(value) != set(AUTHORIZATION_UNSIGNED_FIELDS) | {
        "authorization_payload_sha256"
    }:
        raise BoreasStage2AuthorizationError("authorization exact field set differs")
    supplied = value.get("authorization_payload_sha256")
    if not isinstance(supplied, str) or SHA256_RE.fullmatch(supplied) is None:
        raise BoreasStage2AuthorizationError("authorization self-hash is invalid")
    unsigned = {
        key: child
        for key, child in value.items()
        if key != "authorization_payload_sha256"
    }
    _same(
        hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
        supplied,
        "authorization self-hash",
    )
    _same(value["schema_version"], AUTHORIZATION_SCHEMA, "authorization schema")
    _same(value["self_hash_semantics"], SELF_HASH_SEMANTICS, "self-hash semantics")
    _same(value["authorization_path"], str(expected_path), "authorization path binding")
    _same(value["root_bindings"], root_bindings, "authorization root bindings")
    _validate_recorded_timestamp(value["timestamp_utc"])
    recorded_free = _strict_nonnegative_int(
        value["disk_free_bytes_at_authorization"],
        "recorded authorization free bytes",
    )
    if recorded_free < MINIMUM_START_FREE_BYTES:
        raise BoreasStage2AuthorizationError(
            "recorded authorization disk was below the frozen threshold"
        )

    if not isinstance(allow_exact_closure_publication_resume, bool):
        raise BoreasStage2AuthorizationError(
            "closure publication resume flag must be boolean"
        )
    git_state = (
        _git_state_allowing_exact_closure_publication(
            roots["repository"], roots["stage2_runtime"]
        )
        if allow_exact_closure_publication_resume
        else _git_state(roots["repository"])
    )
    _same(value["branch"], git_state["branch"], "authorization live branch")
    _same(value["commit"], git_state["commit"], "authorization exact live HEAD")
    stage1_evidence_path = _stage1_report_evidence_path(
        roots["stage2_runtime"], create=False
    )
    _same(
        value["stage1_verification_report_path"],
        str(stage1_evidence_path),
        "Stage-1 verification evidence path",
    )
    live = _live_authorities(
        repository=roots["repository"],
        data_root=roots["stage1_data"],
        supplied_stage1_report=stage1_evidence_path,
    )
    expected_unsigned = _authorization_unsigned(
        git_state=git_state,
        roots=root_bindings,
        output=expected_path,
        live=live,
        stage1_report_path=stage1_evidence_path,
        disk_free_bytes=recorded_free,
        timestamp_utc=value["timestamp_utc"],
    )
    _same(unsigned, expected_unsigned, "authorization/live authority projection")
    # A fresh authorization is only written above the frozen start threshold.
    # On a later resume, however, the authenticated 41.999 GB replay allocation
    # is already reflected in live free space.  Requiring the *fresh-start*
    # threshold again would make a correct low-disk run impossible to resume.
    # The production runner independently proves the exact nonsparse replay and
    # ledger before choosing its RESUME gate; otherwise it still applies START.
    # Capability minting therefore enforces the universal runtime reserve here,
    # while the runner owns the mutually exclusive START/RESUME decision.
    live_free = _strict_nonnegative_int(
        int(shutil.disk_usage(roots["monitored_disk"]).free), "runtime live free bytes"
    )
    if live_free < RUNTIME_LOW_DISK_WATERMARK_BYTES:
        raise BoreasStage2AuthorizationError(
            "BLOCKED_INSUFFICIENT_DISK: live free bytes are below the frozen "
            f"runtime watermark ({live_free} < {RUNTIME_LOW_DISK_WATERMARK_BYTES})"
        )
    payload = canonical_json_bytes(value)
    capability = VerifiedStage2Authorization(
        _token=_CAPABILITY_TOKEN,
        _document_bytes=payload,
        repository=roots["repository"],
        stage1_data_root=roots["stage1_data"],
        runtime_root=roots["stage2_runtime"],
        temporary_root=roots["stage2_temporary"],
        monitored_disk_path=roots["monitored_disk"],
        authorization_path=expected_path,
        authorization_file_sha256=hashlib.sha256(payload).hexdigest(),
        no_registration_guard=no_registration_guard,
    )
    object.__setattr__(
        capability, "_formal_verification_token", _FORMAL_VERIFICATION_TOKEN
    )
    if allow_exact_closure_publication_resume:
        # The exact publication-only worktree was already authenticated above.
        # Recheck the immutable capability/root/guard bindings without invoking
        # the normal strict-clean Git gate a second time.
        return capability.assert_operation_live()
    return capability.assert_live()


__all__ = [
    "BoreasStage2AuthorizationError",
    "VerifiedStage2Authorization",
    "build_boreas_v2_stage2_download_authorization",
    "verify_boreas_v2_stage2_download_authorization",
]
