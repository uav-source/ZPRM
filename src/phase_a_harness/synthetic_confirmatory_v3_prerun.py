"""Fail-closed, zero-instantiation entry gates for Confirmatory v3.

The formal v3 runtime is deliberately external to the Git repository.  This
module validates the frozen metadata, plan, Git identity, and external path
contract without importing a snapshot builder, an RNG implementation, or a
registration backend.  In particular, :func:`dry_run_synthetic_confirmatory_v3`
never creates the formal root or any child of it.

The module is also the common gate used by the thin analysis, independent
verification, publication, and artifact-verification command-line wrappers.
It contains no formal executor.  Until a separately frozen executor is
available, a non-dry-run request is rejected after the release-tag and clean
Git gates have passed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAL_RUNTIME_ROOT = Path(
    "/home/lj/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3"
)
FORMAL_BRANCH = "fix/zero-perturbation-v3-qualification-json-native-r3"
FORMAL_PRERUN_TAG = (
    "archive/zero-perturbation-synthetic-confirmatory-v3-"
    "bootstrap-repair-r3-pre-run-pass"
)
DEFAULT_MANIFEST_RELATIVE = Path(
    "frozen_assets/synthetic_confirmatory_formal_manifest_v3_bootstrap_repair_r3.json"
)
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
RUNTIME_ARCHIVE_ROOT = Path("/home/lj/zero_perturbation_runtime_archive")

EXPECTED_SNAPSHOT_COUNT = 595
EXPECTED_TRIAL_COUNT = 1190
EXPECTED_CONDITION_COUNTS = {
    "FULL_NOISE": 525,
    "IDEAL_MATCHED": 35,
    "INDEPENDENT_NOISE_FREE": 35,
}
EXPECTED_BACKEND_COUNTS = {
    "open3d_point_to_plane": 595,
    "pcl_point_to_plane": 595,
}
EXPECTED_SCENE_COUNT = 7
EXPECTED_GEOMETRY_COUNT = 5
EXPECTED_FULL_REALIZATIONS_PER_SCENE_GEOMETRY = 15

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

# These are the qualified, version-independent lifecycle primitives.  Their
# contents are not changed by this thin adapter.  The aggregate binding is the
# canonical SHA-256 of {relative_path: file_sha256}.
LIFECYCLE_CORE_FILES = (
    "src/phase_a_harness/runtime_path_policy.py",
    "src/phase_a_harness/runtime_git_gate.py",
    "src/phase_a_harness/runtime_lifecycle_io.py",
)

DRY_RUN_SCHEMA = "synthetic_confirmatory_v3_zero_instantiation_dry_run_v1"
PATH_CONTRACT_SCHEMA = "synthetic_confirmatory_v3_external_path_contract_v1"


class V3ContractError(ValueError):
    """A frozen v3 metadata, plan, identity, or path contract was violated."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_object(path: str | Path) -> dict[str, Any]:
    """Read one finite JSON object while rejecting duplicate keys."""

    candidate = Path(path)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise V3ContractError(f"duplicate JSON key in {candidate}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V3ContractError(
                    f"non-finite JSON constant in {candidate}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V3ContractError(f"cannot read canonical JSON object: {candidate}") from error
    if type(value) is not dict:
        raise V3ContractError(f"JSON root must be an object: {candidate}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _overlaps(left: Path, right: Path) -> bool:
    return _is_relative_to(left, right) or _is_relative_to(right, left)


def _symlink_components(path: Path) -> list[str]:
    """Return every existing symlink component without following it silently."""

    if not path.is_absolute():
        return [str(path)]
    current = Path(path.anchor)
    found: list[str] = []
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                found.append(str(current))
        except OSError:
            found.append(str(current))
    return found


def _safe_repository_file(repository: Path, relative: str, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise V3ContractError(f"{label} must be a canonical repository-relative path")
    candidate = repository.joinpath(raw)
    normalized = Path(os.path.abspath(os.fspath(candidate)))
    if normalized != candidate:
        raise V3ContractError(f"{label} is not lexically canonical")
    resolved = candidate.resolve(strict=False)
    if not _is_relative_to(resolved, repository):
        raise V3ContractError(f"{label} escapes the harness repository")
    if _symlink_components(candidate):
        raise V3ContractError(f"{label} contains a symlink component")
    if not candidate.is_file():
        raise V3ContractError(f"{label} is not a regular file: {candidate}")
    return candidate


def _validated_manifest_path(repository: Path, manifest_path: str | Path) -> Path:
    raw = Path(manifest_path)
    candidate = raw if raw.is_absolute() else repository / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    resolved = candidate.resolve(strict=False)
    if candidate != resolved:
        raise V3ContractError("v3 manifest path must be canonical and symlink-free")
    if not _is_relative_to(candidate, repository):
        raise V3ContractError("v3 manifest must reside in the harness repository")
    if _symlink_components(candidate):
        raise V3ContractError("v3 manifest path contains a symlink component")
    if not candidate.is_file():
        raise FileNotFoundError(f"v3 frozen manifest is missing: {candidate}")
    return candidate


def _call_contract_loader(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Use the metadata-only contract API when present, with a direct fallback."""

    try:
        module = importlib.import_module(
            "phase_a_harness.synthetic_confirmatory_v3_contract"
        )
    except ModuleNotFoundError as error:
        if error.name != "phase_a_harness.synthetic_confirmatory_v3_contract":
            raise
        return None, "canonical_json_direct"

    loader = getattr(module, "load_v3_contract", None)
    name = "load_v3_contract"
    if loader is None:
        loader = getattr(module, "load_contract", None)
        name = "load_contract"
    if loader is None or not callable(loader):
        return None, "canonical_json_direct"
    loaded = loader(path)
    if type(loaded) is not dict:
        raise V3ContractError(f"{name} must return a plain dict")
    if type(loaded.get("manifest")) is dict:
        loaded = loaded["manifest"]
    return dict(loaded), name


def load_v3_contract_compat(
    manifest_path: str | Path,
    *,
    repository: str | Path,
) -> tuple[Path, dict[str, Any], str]:
    """Load v3 through its public API while retaining direct JSON compatibility."""

    root = Path(repository).resolve()
    path = _validated_manifest_path(root, manifest_path)
    direct = strict_json_object(path)

    payload_sha = direct.get("manifest_payload_sha256")
    if payload_sha is not None:
        if not isinstance(payload_sha, str) or len(payload_sha) != 64:
            raise V3ContractError("v3 manifest payload SHA is malformed")
        payload = {
            key: value
            for key, value in direct.items()
            if key != "manifest_payload_sha256"
        }
        if _canonical_json_sha256(payload) != payload_sha:
            raise V3ContractError("v3 manifest payload SHA mismatch")

    loaded, loader_name = _call_contract_loader(path)
    if loaded is not None and loaded != direct:
        raise V3ContractError("v3 contract loader differs from canonical manifest JSON")
    manifest = direct if loaded is None else loaded
    schema_candidates = (
        manifest.get("manifest_schema"),
        manifest.get("schema_version"),
        manifest.get("artifact_schema"),
    )
    if not any(isinstance(value, str) and "v3" in value.lower() for value in schema_candidates):
        raise V3ContractError("v3 manifest schema identity is missing")
    return path, manifest, loader_name


def lifecycle_core_binding(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    files = {
        relative: _file_sha256(_safe_repository_file(root, relative, relative))
        for relative in LIFECYCLE_CORE_FILES
    }
    return {
        "aggregate_sha256": _canonical_json_sha256(files),
        "files": files,
        "schema_version": "runtime_lifecycle_core_binding_v1",
    }


def _validate_bound_files(
    repository: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = manifest.get("bound_files")
    if type(raw) is not dict or not raw:
        raise V3ContractError("v3 manifest bound_files is missing or empty")
    verified: dict[str, dict[str, Any]] = {}
    path_to_sha: dict[str, str] = {}
    for name, entry in sorted(raw.items()):
        if not isinstance(name, str) or not name or type(entry) is not dict:
            raise V3ContractError("v3 bound_files entry is malformed")
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise V3ContractError(f"v3 bound file metadata is malformed: {name}")
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
            raise V3ContractError(f"v3 bound file SHA is malformed: {name}")
        candidate = _safe_repository_file(repository, relative, f"bound_files.{name}")
        actual = _file_sha256(candidate)
        if actual != expected_sha:
            raise V3ContractError(f"v3 bound file SHA mismatch: {relative}")
        if relative in path_to_sha and path_to_sha[relative] != expected_sha:
            raise V3ContractError(f"conflicting v3 bound-file entries: {relative}")
        path_to_sha[relative] = expected_sha
        verified[name] = {
            "path": relative,
            "sha256": actual,
            "size_bytes": candidate.stat().st_size,
        }

    lifecycle = lifecycle_core_binding(repository)
    expected_aggregate = manifest.get("runtime_lifecycle_core_sha256")
    aggregate_bound = (
        expected_aggregate is not None
        and expected_aggregate == lifecycle["aggregate_sha256"]
    )
    individual_bound = all(
        path_to_sha.get(relative) == sha
        for relative, sha in lifecycle["files"].items()
    )
    if expected_aggregate is not None and not aggregate_bound:
        raise V3ContractError("v3 runtime lifecycle aggregate SHA mismatch")
    if not aggregate_bound and not individual_bound:
        raise V3ContractError("v3 manifest does not bind the qualified lifecycle core")
    lifecycle = {
        **lifecycle,
        "aggregate_manifest_binding_present": expected_aggregate is not None,
        "aggregate_manifest_binding_pass": aggregate_bound,
        "individual_bound_file_pass": individual_bound,
        "RUNTIME_LIFECYCLE_CORE_BINDING_PASS": True,
    }
    return verified, lifecycle


def _bound_asset_path(
    repository: Path,
    manifest: Mapping[str, Any],
    names: Sequence[str],
    top_level_names: Sequence[str],
) -> Path:
    bound = manifest.get("bound_files")
    assert type(bound) is dict
    candidates: list[str] = []
    for name in names:
        entry = bound.get(name)
        if type(entry) is dict and isinstance(entry.get("path"), str):
            candidates.append(entry["path"])
    for name in top_level_names:
        value = manifest.get(name)
        if isinstance(value, str):
            candidates.append(value)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise V3ContractError(
            f"v3 manifest must bind exactly one asset path for {names[0]}"
        )
    return _safe_repository_file(repository, unique[0], names[0])


def _canonical_integer(value: str, label: str) -> int:
    if value == "" or value.strip() != value:
        raise V3ContractError(f"{label} is not a canonical integer")
    try:
        result = int(value)
    except ValueError as error:
        raise V3ContractError(f"{label} is not an integer") from error
    if str(result) != value:
        raise V3ContractError(f"{label} is not a canonical integer")
    return result


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise V3ContractError(f"v3 plan CSV schema mismatch: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise V3ContractError(f"cannot read v3 plan CSV: {path}") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise V3ContractError(f"v3 plan CSV row has missing or excess fields: {path}")
    return rows


def _typed_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_read_csv(path, SNAPSHOT_FIELDS), start=2):
        measurement = row["measurement_seed"]
        rows.append(
            {
                "planned_snapshot_id": row["planned_snapshot_id"],
                "scene_variant": row["scene_variant"],
                "condition": row["condition"],
                "geometry_seed": _canonical_integer(
                    row["geometry_seed"], f"snapshot row {index} geometry_seed"
                ),
                "measurement_seed": (
                    None
                    if measurement == ""
                    else _canonical_integer(
                        measurement, f"snapshot row {index} measurement_seed"
                    )
                ),
                "repeat_index": _canonical_integer(
                    row["repeat_index"], f"snapshot row {index} repeat_index"
                ),
                "planned_backend_count": _canonical_integer(
                    row["planned_backend_count"],
                    f"snapshot row {index} planned_backend_count",
                ),
                "replicate_semantics": row["replicate_semantics"],
            }
        )
    return rows


def _typed_trial_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_read_csv(path, TRIAL_FIELDS), start=2):
        measurement = row["measurement_seed"]
        rows.append(
            {
                "planned_trial_id": row["planned_trial_id"],
                "planned_snapshot_id": row["planned_snapshot_id"],
                "scene_variant": row["scene_variant"],
                "condition": row["condition"],
                "geometry_seed": _canonical_integer(
                    row["geometry_seed"], f"trial row {index} geometry_seed"
                ),
                "measurement_seed": (
                    None
                    if measurement == ""
                    else _canonical_integer(
                        measurement, f"trial row {index} measurement_seed"
                    )
                ),
                "repeat_index": _canonical_integer(
                    row["repeat_index"], f"trial row {index} repeat_index"
                ),
                "backend": row["backend"],
            }
        )
    return rows


def audit_v3_plan(
    snapshot_path: str | Path,
    trial_path: str | Path,
) -> dict[str, Any]:
    """Statically audit the exact 595/1190 plan without touching formal state."""

    snapshots = _typed_snapshot_rows(Path(snapshot_path))
    trials = _typed_trial_rows(Path(trial_path))
    snapshot_ids = [row["planned_snapshot_id"] for row in snapshots]
    trial_ids = [row["planned_trial_id"] for row in trials]
    if any(not value for value in snapshot_ids + trial_ids):
        raise V3ContractError("v3 plan contains an empty planned identity")

    snapshot_by_id = {row["planned_snapshot_id"]: row for row in snapshots}
    duplicate_snapshot_count = len(snapshot_ids) - len(snapshot_by_id)
    duplicate_trial_count = len(trial_ids) - len(set(trial_ids))
    condition_counts = Counter(row["condition"] for row in snapshots)
    backend_counts = Counter(row["backend"] for row in trials)
    scenes = {row["scene_variant"] for row in snapshots}
    geometry_values = {row["geometry_seed"] for row in snapshots}

    semantic_violation_count = 0
    independent_pseudoreplication_plan_count = 0
    ideal_groups: Counter[tuple[str, int]] = Counter()
    independent_groups: Counter[tuple[str, int]] = Counter()
    full_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in snapshots:
        key = (row["scene_variant"], row["geometry_seed"])
        if (
            row["planned_backend_count"] != 2
            or row["condition"] not in EXPECTED_CONDITION_COUNTS
            or not row["scene_variant"]
        ):
            semantic_violation_count += 1
        if row["condition"] == "IDEAL_MATCHED":
            ideal_groups[key] += 1
            if (
                row["measurement_seed"] is not None
                or row["repeat_index"] != 0
                or row["replicate_semantics"] != "ONE_CONTROL_INPUT"
            ):
                semantic_violation_count += 1
        elif row["condition"] == "INDEPENDENT_NOISE_FREE":
            independent_groups[key] += 1
            invalid = row["measurement_seed"] is not None or row["repeat_index"] != 0
            independent_pseudoreplication_plan_count += int(invalid)
            if (
                invalid
                or row["replicate_semantics"]
                != "ONE_DETERMINISTIC_INDEPENDENT_INPUT"
            ):
                semantic_violation_count += 1
        elif row["condition"] == "FULL_NOISE":
            full_groups[key].append(row)
            if (
                row["measurement_seed"] is None
                or row["repeat_index"] not in range(5)
                or row["replicate_semantics"]
                != "FIFTEEN_STOCHASTIC_INPUTS_PER_SCENE_GEOMETRY"
            ):
                semantic_violation_count += 1

    expected_blocks = {
        (scene, geometry) for scene in scenes for geometry in geometry_values
    }
    independent_pseudoreplication_plan_count += sum(
        max(0, count - 1) for count in independent_groups.values()
    )
    ideal_block_pass = bool(
        len(scenes) == EXPECTED_SCENE_COUNT
        and len(geometry_values) == EXPECTED_GEOMETRY_COUNT
        and set(ideal_groups) == expected_blocks
        and all(count == 1 for count in ideal_groups.values())
    )
    independent_block_pass = bool(
        set(independent_groups) == expected_blocks
        and all(count == 1 for count in independent_groups.values())
    )
    full_block_pass = bool(
        set(full_groups) == expected_blocks
        and all(
            len(rows) == EXPECTED_FULL_REALIZATIONS_PER_SCENE_GEOMETRY
            and len(
                {
                    (row["measurement_seed"], row["repeat_index"])
                    for row in rows
                }
            )
            == EXPECTED_FULL_REALIZATIONS_PER_SCENE_GEOMETRY
            and len({row["measurement_seed"] for row in rows}) == 3
            and {row["repeat_index"] for row in rows} == set(range(5))
            for rows in full_groups.values()
        )
    )

    trial_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orphan_trial_count = 0
    trial_identity_mismatch_count = 0
    trial_metadata_mismatch_count = 0
    metadata_names = (
        "scene_variant",
        "condition",
        "geometry_seed",
        "measurement_seed",
        "repeat_index",
    )
    for row in trials:
        snapshot_id = row["planned_snapshot_id"]
        trial_groups[snapshot_id].append(row)
        snapshot = snapshot_by_id.get(snapshot_id)
        if snapshot is None:
            orphan_trial_count += 1
            continue
        if row["planned_trial_id"] != f"{snapshot_id}::{row['backend']}":
            trial_identity_mismatch_count += 1
        if any(row[name] != snapshot[name] for name in metadata_names):
            trial_metadata_mismatch_count += 1
    pairing_violation_count = orphan_trial_count + trial_metadata_mismatch_count
    expected_backends = set(EXPECTED_BACKEND_COUNTS)
    for snapshot_id in snapshot_ids:
        rows = trial_groups.get(snapshot_id, [])
        if len(rows) != 2 or {row["backend"] for row in rows} != expected_backends:
            pairing_violation_count += 1

    native_trial_count = sum(
        count for backend, count in backend_counts.items() if backend not in expected_backends
    )
    count_pass = bool(
        len(snapshots) == EXPECTED_SNAPSHOT_COUNT
        and len(trials) == EXPECTED_TRIAL_COUNT
        and dict(condition_counts) == EXPECTED_CONDITION_COUNTS
        and dict(backend_counts) == EXPECTED_BACKEND_COUNTS
        and semantic_violation_count == 0
        and ideal_block_pass
        and full_block_pass
    )
    uniqueness_pass = bool(
        duplicate_snapshot_count == 0
        and duplicate_trial_count == 0
        and trial_identity_mismatch_count == 0
    )
    pairing_pass = pairing_violation_count == 0
    independent_pass = bool(
        independent_pseudoreplication_plan_count == 0
        and independent_block_pass
    )
    report = {
        "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": independent_pass,
        "CONFIRMATORY_PLAN_COUNT_PASS": count_pass,
        "CONFIRMATORY_PLAN_PAIRING_PASS": pairing_pass,
        "CONFIRMATORY_PLAN_UNIQUENESS_PASS": uniqueness_pass,
        "V3_PLAN_AUDIT_PASS": bool(
            count_pass and uniqueness_pass and pairing_pass and independent_pass
        ),
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "duplicate_snapshot_count": duplicate_snapshot_count,
        "duplicate_trial_count": duplicate_trial_count,
        "full_noise_block_pass": full_block_pass,
        "full_noise_replicates_per_scene_geometry": (
            EXPECTED_FULL_REALIZATIONS_PER_SCENE_GEOMETRY
            if full_block_pass
            else None
        ),
        "ideal_block_pass": ideal_block_pass,
        "independent_block_pass": independent_block_pass,
        "independent_pseudoreplication_plan_count": (
            independent_pseudoreplication_plan_count
        ),
        "native_trial_count": native_trial_count,
        "orphan_trial_count": orphan_trial_count,
        "pairing_violation_count": pairing_violation_count,
        "planned_snapshot_count": len(snapshots),
        "planned_snapshot_identity_sha256": _canonical_json_sha256(snapshots),
        "planned_snapshot_unique_count": len(set(snapshot_ids)),
        "planned_trial_count": len(trials),
        "planned_trial_identity_sha256": _canonical_json_sha256(trials),
        "planned_trial_unique_count": len(set(trial_ids)),
        "scene_count": len(scenes),
        "semantic_violation_count": semantic_violation_count,
        "trial_identity_mismatch_count": trial_identity_mismatch_count,
        "trial_metadata_mismatch_count": trial_metadata_mismatch_count,
    }
    if report["V3_PLAN_AUDIT_PASS"] is not True:
        raise V3ContractError("v3 frozen plan audit failed")
    return report


def load_v3_plan_rows(
    snapshot_path: str | Path,
    trial_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return typed plan rows after the complete static audit has passed."""

    audit_v3_plan(snapshot_path, trial_path)
    return (
        _typed_snapshot_rows(Path(snapshot_path)),
        _typed_trial_rows(Path(trial_path)),
    )


def _manifest_value(
    manifest: Mapping[str, Any], names: Sequence[str], label: str
) -> Any:
    values = [manifest[name] for name in names if name in manifest]
    if not values:
        raise V3ContractError(f"v3 manifest is missing {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise V3ContractError(f"v3 manifest contains conflicting {label} values")
    return first


def _authorization_value(manifest: Mapping[str, Any]) -> bool:
    names = (
        "formal_execution_authorized",
        "CONFIRMATORY_V3_RUN_AUTHORIZED",
        "confirmatory_v3_run_authorized",
    )
    present = [manifest[name] for name in names if name in manifest]
    if present:
        if any(type(value) is not bool for value in present) or any(
            value != present[0] for value in present[1:]
        ):
            raise V3ContractError("v3 formal authorization is malformed")
        return present[0]
    state = manifest.get("formal_run_authorization_state")
    if state == "PENDING_VERIFIED_PRE_RUN_FINAL_DECISION":
        return False
    raise V3ContractError("v3 manifest authorization state is missing")


def _identity_value(
    manifest: Mapping[str, Any], names: Sequence[str], label: str
) -> str:
    value = _manifest_value(manifest, names, label)
    if not isinstance(value, str) or not value or any(ch in value for ch in "\0\n\r"):
        raise V3ContractError(f"v3 {label} is invalid")
    return value


def manifest_git_identity(
    manifest: Mapping[str, Any],
    *,
    repository: str | Path,
    require_release_tag: bool,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    commit_values = [
        manifest[name]
        for name in (
            "formal_commit",
            "formal_commit_sha",
            "formal_git_commit",
            "repository_commit",
        )
        if name in manifest
    ]
    if commit_values:
        commit = str(commit_values[0])
        if any(value != commit_values[0] for value in commit_values[1:]):
            raise V3ContractError("v3 manifest contains conflicting formal commits")
    else:
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise V3ContractError("cannot resolve v3 Git HEAD") from error
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise V3ContractError("v3 formal commit must be a lowercase full SHA")
    branch = _identity_value(
        manifest, ("formal_branch", "expected_branch"), "formal branch"
    )
    tag = _identity_value(
        manifest,
        (
            "formal_release_tag",
            "release_tag",
            "formal_pre_run_tag",
            "pre_run_tag",
            "expected_tag",
            "expected_release_tag",
        ),
        "final release tag" if require_release_tag else "frozen Git tag",
    )
    if tag.startswith("refs/"):
        raise V3ContractError("v3 frozen Git tag must not start with refs/")
    tag_exists = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).returncode == 0
    if require_release_tag and not tag_exists:
        raise V3ContractError("v3 final release tag does not exist")
    return {
        "branch": branch,
        "commit": commit,
        "tag": tag,
        "tag_enforced": bool(require_release_tag or tag_exists),
    }


def _formal_layout(root: Path) -> dict[str, Path]:
    return {
        "run_root": root,
        "snapshot_cache": root / "snapshot_cache",
        "snapshot_lock": root / "snapshot_lock.json",
        "raw_results": root / "raw_results",
        "event_log": root / "event_logs",
        "backend_temporary": root / "backend_tmp",
        "analysis": root / "analysis",
        "verification": root / "verification",
        "publisher_staging": root / "publisher_staging",
        "artifact_staging": root / "artifact_staging",
        "temporary_inventory": root / "working_inventory",
    }


def qualify_v3_formal_runtime_path(
    runtime_root: str | Path,
    *,
    repository: str | Path,
    require_absent: bool,
) -> dict[str, Any]:
    """Validate the exact external v3 layout without creating any path."""

    raw = Path(runtime_root)
    if not raw.is_absolute():
        raise V3ContractError("v3 formal runtime root must be absolute")
    normalized = Path(os.path.abspath(os.fspath(raw)))
    if raw != normalized or normalized != FORMAL_RUNTIME_ROOT:
        raise V3ContractError(
            f"v3 formal runtime root must be exactly {FORMAL_RUNTIME_ROOT}"
        )
    resolved = raw.resolve(strict=False)
    symlinks = _symlink_components(raw)
    if resolved != raw or symlinks:
        raise V3ContractError("v3 formal runtime root must be canonical and symlink-free")

    repository_root = Path(repository).resolve()
    forbidden = {
        "harness_repository": repository_root,
        "source_repository": SOURCE_REPOSITORY.resolve(),
        "runtime_archive": RUNTIME_ARCHIVE_ROOT.resolve(strict=False),
        "runtime_qualification": Path(
            "/home/lj/zero_perturbation_runtime/qualification"
        ).resolve(strict=False),
    }
    overlap = [name for name, path in forbidden.items() if _overlaps(raw, path)]
    if overlap:
        raise V3ContractError(
            "v3 formal runtime root overlaps forbidden roots: " + ", ".join(overlap)
        )
    if require_absent and raw.exists():
        raise FileExistsError("v3 zero-instantiation dry-run requires absent formal root")

    layout = _formal_layout(raw)
    path_checks: dict[str, dict[str, Any]] = {}
    for name, path in layout.items():
        children = _symlink_components(path)
        inside = _is_relative_to(path, raw)
        canonical = path.resolve(strict=False) == path
        if not path.is_absolute() or not inside or not canonical or children:
            raise V3ContractError(f"v3 formal layout path failed: {name}")
        if require_absent and path.exists():
            raise FileExistsError(f"v3 dry-run layout path already exists: {path}")
        path_checks[name] = {
            "absolute": path.is_absolute(),
            "canonical": canonical,
            "inside_run_root": inside,
            "path": str(path),
            "symlink_component_count": len(children),
        }

    leaf_names = [name for name in layout if name != "run_root"]
    leaf_overlap: list[list[str]] = []
    for index, left_name in enumerate(leaf_names):
        for right_name in leaf_names[index + 1 :]:
            if _overlaps(layout[left_name], layout[right_name]):
                leaf_overlap.append([left_name, right_name])
    if leaf_overlap:
        raise V3ContractError("v3 formal layout contains overlapping mutable leaves")

    return {
        "FORMAL_RUNTIME_PATH_CONTRACT_PASS": True,
        "all_paths_absolute": True,
        "all_paths_canonical": True,
        "all_paths_inside_run_root": True,
        "all_paths_without_symlink_components": True,
        "forbidden_root_overlap_count": 0,
        "layout": {name: str(path) for name, path in layout.items()},
        "mutable_leaf_overlap_count": 0,
        "path_checks": path_checks,
        "require_absent": require_absent,
        "runtime_root": str(raw),
        "schema_version": PATH_CONTRACT_SCHEMA,
        "symlink_component_count": 0,
    }


def _validate_manifest_runtime_identity(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    runtime_root: Path,
    workers: int,
) -> None:
    expected_run_id = _identity_value(
        manifest, ("formal_run_id", "run_id"), "formal run ID"
    )
    expected_root = _identity_value(
        manifest,
        ("formal_runtime_root", "formal_output_dir", "runtime_root"),
        "formal runtime root",
    )
    expected_workers = _manifest_value(
        manifest, ("formal_workers", "workers"), "formal workers"
    )
    if (
        run_id != expected_run_id
        or expected_root != str(FORMAL_RUNTIME_ROOT)
        or runtime_root != FORMAL_RUNTIME_ROOT
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
        or workers != expected_workers
    ):
        raise V3ContractError("v3 formal invocation differs from frozen manifest")
    for name, expected in (
        ("planned_snapshot_count", EXPECTED_SNAPSHOT_COUNT),
        ("planned_trial_count", EXPECTED_TRIAL_COUNT),
    ):
        if manifest.get(name) != expected:
            raise V3ContractError(f"v3 manifest {name} mismatch")
    backend_count = manifest.get("backend_count")
    if backend_count is None and type(manifest.get("backend_bindings")) is dict:
        backend_count = sum(
            name in manifest["backend_bindings"] for name in ("open3d", "pcl")
        )
    if backend_count != 2:
        raise V3ContractError("v3 manifest backend count mismatch")


def verify_v3_prerun_authorization(
    repository: str | Path,
    manifest: Mapping[str, Any],
    *,
    require_runtime_absent: bool,
) -> dict[str, Any]:
    """Authenticate the final pre-run package that grants formal authority."""

    root = Path(repository).resolve()
    binding = manifest.get("authorization_binding")
    if type(binding) is not dict:
        raise V3ContractError("v3 authorization binding is missing")
    relative = binding.get("pre_run_final_decision_path")
    if not isinstance(relative, str):
        raise V3ContractError("v3 pre-run decision path binding is missing")
    decision_path = _safe_repository_file(root, relative, "pre-run final decision")
    artifact = decision_path.parent
    from .synthetic_confirmatory_v3_bootstrap_repair_r3_artifact_verifier import (
        verify_bootstrap_repair_r3_prerun_artifact,
    )

    live = verify_bootstrap_repair_r3_prerun_artifact(
        artifact,
        require_formal_runtime_absent=require_runtime_absent,
    )
    if live.get("PRE_RUN_ARTIFACT_VERIFICATION_PASS") is not True:
        raise PermissionError("v3 final pre-run artifact verification failed")
    decision = strict_json_object(decision_path)
    required = {
        "CONFIRMATORY_V3_RUN_AUTHORIZED": True,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "SYNTHETIC_CONFIRMATORY_V3_PRE_RUN_QUALIFICATION_PASS": True,
        "SYNTHETIC_CONFIRMATORY_V3_BOOTSTRAP_REPAIR_R3_PRE_RUN_QUALIFICATION_PASS": True,
    }
    if any(decision.get(name) != value for name, value in required.items()):
        raise PermissionError("v3 final pre-run authorization decision changed")
    if (
        binding.get("required_authorization_field")
        != "CONFIRMATORY_V3_RUN_AUTHORIZED"
        or binding.get("required_authorization_value") is not True
        or binding.get("expected_release_tag")
        != manifest.get("expected_release_tag")
    ):
        raise V3ContractError("v3 manifest authorization binding is inconsistent")
    return {
        "FORMAL_PRERUN_ARTIFACT_BINDING_PASS": True,
        "artifact_path": str(artifact),
        "artifact_verification": live,
        "decision_path": str(decision_path),
        "decision_sha256": _file_sha256(decision_path),
    }


def validate_v3_frozen_contract(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    require_authorized: bool,
    require_release_tag: bool,
    require_runtime_absent: bool,
    git_checkpoint: str,
) -> dict[str, Any]:
    """Authenticate all static v3 inputs, Git identity, plan, and path policy."""

    root = Path(repository).resolve()
    try:
        current_commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PermissionError("cannot establish the formal Git identity") from error
    from .runtime_git_gate import verify_runtime_git_gate

    # This preliminary gate intentionally precedes manifest, plan, authorization
    # artifact, and runtime-inventory reads.  The release tag itself binds the
    # final commit; dry-run uses the same branch/clean gate without requiring it.
    git_gate = verify_runtime_git_gate(
        root,
        expected_commit=current_commit,
        expected_branch=FORMAL_BRANCH,
        expected_tag=(FORMAL_PRERUN_TAG if require_release_tag else None),
        checkpoint=git_checkpoint,
    )
    manifest_file, manifest, loader_name = load_v3_contract_compat(
        manifest_path, repository=root
    )
    declared_authorization = _authorization_value(manifest)
    if require_authorized:
        authorization_binding = verify_v3_prerun_authorization(
            root,
            manifest,
            require_runtime_absent=require_runtime_absent,
        )
        authorization = True
    else:
        if declared_authorization is not False:
            raise PermissionError("v3 dry-run requires pending/false authorization")
        authorization_binding = None
        authorization = False

    runtime = Path(runtime_root)
    _validate_manifest_runtime_identity(
        manifest,
        run_id=run_id,
        runtime_root=runtime,
        workers=workers,
    )
    bound_files, lifecycle = _validate_bound_files(root, manifest)
    snapshots_path = _bound_asset_path(
        root,
        manifest,
        ("planned_snapshots", "snapshot_plan"),
        ("planned_snapshots_path", "snapshot_plan_path"),
    )
    trials_path = _bound_asset_path(
        root,
        manifest,
        ("planned_trials", "trial_plan"),
        ("planned_trials_path", "trial_plan_path"),
    )
    plan = audit_v3_plan(snapshots_path, trials_path)
    if manifest.get("condition_snapshot_counts") not in (
        None,
        EXPECTED_CONDITION_COUNTS,
    ):
        raise V3ContractError("v3 manifest condition counts mismatch")
    if manifest.get("backend_trial_counts") not in (
        None,
        EXPECTED_BACKEND_COUNTS,
    ):
        raise V3ContractError("v3 manifest backend counts mismatch")

    path_contract = qualify_v3_formal_runtime_path(
        runtime,
        repository=root,
        require_absent=require_runtime_absent,
    )
    manifest_runtime_bindings = {
        "snapshot_cache": "snapshot_cache_path",
        "snapshot_lock": "snapshot_lock_path",
        "raw_results": "raw_results_path",
        "event_log": "event_log_path",
        "backend_temporary": "backend_temporary_path",
        "analysis": "analysis_path",
        "verification": "verification_path",
        "publisher_staging": "publisher_staging_path",
        "artifact_staging": "artifact_staging_path",
        "temporary_inventory": "temporary_inventory_path",
    }
    for layout_name, manifest_name in manifest_runtime_bindings.items():
        if manifest.get(manifest_name) != path_contract["layout"][layout_name]:
            raise V3ContractError(
                f"v3 manifest runtime path mismatch: {manifest_name}"
            )
    identity = manifest_git_identity(
        manifest,
        repository=root,
        require_release_tag=require_release_tag,
    )
    if (
        identity["commit"] != current_commit
        or identity["branch"] != FORMAL_BRANCH
        or identity["tag"] != FORMAL_PRERUN_TAG
    ):
        raise V3ContractError("v3 manifest Git identity differs from the early gate")
    return {
        "FORMAL_GIT_GATE_PASS": git_gate.get("RUNTIME_GIT_GATE_PASS") is True,
        "FORMAL_PRERUN_ARTIFACT_BINDING_PASS": (
            authorization_binding is not None
            and authorization_binding["FORMAL_PRERUN_ARTIFACT_BINDING_PASS"] is True
        ) if require_authorized else True,
        "V3_FROZEN_CONTRACT_PASS": True,
        "authorization": authorization,
        "authorization_binding": authorization_binding,
        "bound_file_count": len(bound_files),
        "bound_files": bound_files,
        "contract_loader": loader_name,
        "git_gate": git_gate,
        "git_identity": identity,
        "lifecycle_core_binding": lifecycle,
        "manifest": manifest,
        "manifest_path": str(manifest_file),
        "manifest_sha256": _file_sha256(manifest_file),
        "plan_audit": plan,
        "planned_snapshots_path": str(snapshots_path),
        "planned_snapshots_sha256": _file_sha256(snapshots_path),
        "planned_trials_path": str(trials_path),
        "planned_trials_sha256": _file_sha256(trials_path),
        "runtime_path_contract": path_contract,
    }


def dry_run_synthetic_confirmatory_v3(
    *,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path = FORMAL_RUNTIME_ROOT,
    workers: int,
    repository: str | Path,
) -> dict[str, Any]:
    """Perform the formal metadata dry-run while proving all side effects zero."""

    formal = Path(runtime_root)
    existed_before = formal.exists()
    if existed_before:
        raise FileExistsError("v3 zero-instantiation dry-run requires absent formal root")
    imported_before = set(sys.modules)
    validated = validate_v3_frozen_contract(
        repository=repository,
        manifest_path=manifest_path,
        run_id=run_id,
        runtime_root=formal,
        workers=workers,
        require_authorized=False,
        require_release_tag=False,
        require_runtime_absent=True,
        git_checkpoint="V3_FORMAL_DRY_RUN_GIT_GATE",
    )
    existed_after = formal.exists()
    if existed_after:
        raise RuntimeError("v3 dry-run created the forbidden formal runtime root")

    newly_imported = sorted(set(sys.modules) - imported_before)
    forbidden_prefixes = (
        "open3d",
        "numpy.random",
        "phase_a_harness.full_synthetic_snapshot_builder",
        "phase_a_harness.full_synthetic_backend_execution",
        "phase_a_harness.open3d_backend",
        "phase_a_harness.pcl_backend",
    )
    forbidden_imports = [
        name
        for name in newly_imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    if forbidden_imports:
        raise RuntimeError(
            "v3 dry-run imported execution-capable modules: "
            + ", ".join(forbidden_imports)
        )

    plan = validated["plan_audit"]
    result = {
        "CONFIRMATORY_BACKEND_EXECUTION_COUNT": 0,
        "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
        "CONFIRMATORY_SNAPSHOT_GENERATION_COUNT": 0,
        "CONFIRMATORY_TRIAL_RESULT_COUNT": 0,
        "FORMAL_ATTEMPT_EVENT_COUNT": 0,
        "FORMAL_BACKEND_EXECUTION_COUNT": 0,
        "FORMAL_RNG_INSTANTIATION_COUNT": 0,
        "FORMAL_SNAPSHOT_CONSTRUCTION_COUNT": 0,
        "FORMAL_TRIAL_RESULT_COUNT": 0,
        "NATIVE_EXECUTION_COUNT": 0,
        "V3_BACKEND_EXECUTION_COUNT": 0,
        "V3_FORMAL_RUNTIME_ROOT_NOT_CREATED": True,
        "V3_RNG_INSTANTIATION_COUNT": 0,
        "V3_SEED_ACCESS_COUNT": 0,
        "V3_SNAPSHOT_CONSTRUCTION_COUNT": 0,
        "V3_STARTED_EVENT_COUNT": 0,
        "V3_TRIAL_RESULT_COUNT": 0,
        "SYNTHETIC_CONFIRMATORY_V3_DRY_RUN_PASS": bool(
            validated["V3_FROZEN_CONTRACT_PASS"] is True
            and validated["FORMAL_GIT_GATE_PASS"] is True
            and plan["V3_PLAN_AUDIT_PASS"] is True
            and plan["native_trial_count"] == 0
            and not existed_before
            and not existed_after
            and not forbidden_imports
        ),
        "SYNTHETIC_CONFIRMATORY_V3_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_V3_PASS": "NOT_EVALUATED",
        "attempt_started_event_count": 0,
        "backend_trial_counts": plan["backend_trial_counts"],
        "backend_trial_counts_canonical": {
            "Native": 0,
            "Open3D": plan["backend_trial_counts"].get("open3d_point_to_plane", 0),
            "PCL": plan["backend_trial_counts"].get("pcl_point_to_plane", 0),
        },
        "condition_snapshot_counts": plan["condition_snapshot_counts"],
        "condition_counts": plan["condition_snapshot_counts"],
        "contract_loader": validated["contract_loader"],
        "formal_root_created": False,
        "formal_root_existed_after": existed_after,
        "formal_root_existed_before": existed_before,
        "formal_runtime_root": str(formal),
        "forbidden_execution_module_import_count": len(forbidden_imports),
        "forbidden_execution_module_imports": forbidden_imports,
        "git_gate": validated["git_gate"],
        "lifecycle_core_binding": validated["lifecycle_core_binding"],
        "manifest_path": validated["manifest_path"],
        "manifest_sha256": validated["manifest_sha256"],
        "formal_manifest_sha256": validated["manifest_sha256"],
        "native_trial_count": plan["native_trial_count"],
        "open3d_trial_count": plan["backend_trial_counts"].get(
            "open3d_point_to_plane", 0
        ),
        "pairing_violation_count": plan["pairing_violation_count"],
        "pcl_trial_count": plan["backend_trial_counts"].get(
            "pcl_point_to_plane", 0
        ),
        "plan_audit": plan,
        "planned_snapshot_count": plan["planned_snapshot_count"],
        "planned_trial_count": plan["planned_trial_count"],
        "unique_snapshot_count": plan["planned_snapshot_unique_count"],
        "unique_trial_count": plan["planned_trial_unique_count"],
        "duplicate_snapshot_count": plan["duplicate_snapshot_count"],
        "duplicate_trial_count": plan["duplicate_trial_count"],
        "independent_pseudoreplication_count": plan[
            "independent_pseudoreplication_plan_count"
        ],
        "run_id": run_id,
        "runtime_path_contract": validated["runtime_path_contract"],
        "schema_version": DRY_RUN_SCHEMA,
        "workers": workers,
    }
    # Canonical qualifier aliases.  The scientific plan continues to retain
    # the exact backend schema names in ``backend_trial_counts`` above.
    result["backend_trial_counts"] = {
        "Native": 0,
        "Open3D": plan["backend_trial_counts"].get("open3d_point_to_plane", 0),
        "PCL": plan["backend_trial_counts"].get("pcl_point_to_plane", 0),
    }
    result["V3_DRY_RUN_PASS"] = result[
        "SYNTHETIC_CONFIRMATORY_V3_DRY_RUN_PASS"
    ]
    if result["SYNTHETIC_CONFIRMATORY_V3_DRY_RUN_PASS"] is not True:
        raise RuntimeError("v3 zero-instantiation dry-run failed")
    return result


def guard_synthetic_confirmatory_v3_execution(
    *,
    manifest_path: str | Path,
    run_id: str,
    runtime_root: str | Path,
    workers: int,
    repository: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Authenticate the final release before an execution-capable import."""

    if resume is not True:
        raise PermissionError("formal v3 execution requires --resume")
    return validate_v3_frozen_contract(
        repository=repository,
        manifest_path=manifest_path,
        run_id=run_id,
        runtime_root=runtime_root,
        workers=workers,
        require_authorized=True,
        require_release_tag=True,
        require_runtime_absent=False,
        git_checkpoint="V3_FORMAL_EXECUTION_ENTRY_GIT_GATE",
    )


def validate_v3_postrun_entry(
    *,
    repository: str | Path,
    manifest_path: str | Path,
    runtime_root: str | Path,
    checkpoint: str,
) -> dict[str, Any]:
    """Gate thin post-run CLIs and fail before science imports if data is absent."""

    root = Path(repository).resolve()
    manifest_file, manifest, _loader = load_v3_contract_compat(
        manifest_path, repository=root
    )
    run_id = _identity_value(manifest, ("formal_run_id", "run_id"), "formal run ID")
    workers = _manifest_value(manifest, ("formal_workers", "workers"), "formal workers")
    report = validate_v3_frozen_contract(
        repository=root,
        manifest_path=manifest_file,
        run_id=run_id,
        runtime_root=runtime_root,
        workers=workers,
        require_authorized=True,
        require_release_tag=True,
        require_runtime_absent=False,
        git_checkpoint=checkpoint,
    )
    runtime = Path(runtime_root)
    if not runtime.is_dir():
        raise FileNotFoundError(
            "v3 formal runtime data is absent; post-run entry remains fail-closed"
        )
    return report


def require_path_inside_formal_root(
    path: str | Path,
    *,
    allow_existing: bool,
) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise V3ContractError("v3 formal output/evidence path must be absolute")
    candidate = Path(os.path.abspath(os.fspath(raw)))
    if candidate != raw or not _is_relative_to(candidate, FORMAL_RUNTIME_ROOT):
        raise V3ContractError("v3 formal output/evidence path escapes the formal root")
    if candidate.resolve(strict=False) != candidate or _symlink_components(candidate):
        raise V3ContractError("v3 formal output/evidence path is noncanonical or symlinked")
    if not allow_existing and candidate.exists():
        raise FileExistsError(f"refusing to replace v3 formal evidence: {candidate}")
    return candidate


def assert_cli_isolation() -> None:
    """Apply the same Python/source-repository isolation to every v3 CLI."""

    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("Synthetic Confirmatory v3 requires PYTHONNOUSERSITE=1")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        value
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    for entry in entries:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        if candidate == source or source in candidate.parents or candidate in source.parents:
            raise PermissionError("Python search path reaches the source repository")


__all__ = [
    "DEFAULT_MANIFEST_RELATIVE",
    "DRY_RUN_SCHEMA",
    "EXPECTED_BACKEND_COUNTS",
    "EXPECTED_CONDITION_COUNTS",
    "EXPECTED_SNAPSHOT_COUNT",
    "EXPECTED_TRIAL_COUNT",
    "FORMAL_RUNTIME_ROOT",
    "LIFECYCLE_CORE_FILES",
    "V3ContractError",
    "assert_cli_isolation",
    "audit_v3_plan",
    "dry_run_synthetic_confirmatory_v3",
    "guard_synthetic_confirmatory_v3_execution",
    "lifecycle_core_binding",
    "load_v3_plan_rows",
    "load_v3_contract_compat",
    "manifest_git_identity",
    "qualify_v3_formal_runtime_path",
    "require_path_inside_formal_root",
    "strict_json_object",
    "validate_v3_frozen_contract",
    "validate_v3_postrun_entry",
]
