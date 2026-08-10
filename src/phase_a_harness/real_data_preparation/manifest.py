"""Immutable small-artifact manifest and independent verification."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from phase_a_harness.real_data_protocol import SNAPSHOT_SELECTION_FIELDS

from .io import compact_sha256, sha256_file


BACKEND_FILE_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
OPEN3D_PARAMETERS_SHA256 = "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
PCL_PARAMETERS_SHA256 = "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"


class ManifestVerificationError(RuntimeError):
    """The preregistration manifest or its inventory is inconsistent."""


def _safe_artifact_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts or "." in posix.parts:
        raise ManifestVerificationError(f"unsafe manifest path: {relative}")
    path = root.joinpath(*posix.parts)
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ManifestVerificationError(f"manifest path escapes root: {relative}")
    if path.is_symlink():
        raise ManifestVerificationError(f"manifest artifact is a symlink: {relative}")
    return path


def artifact_inventory(root: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        try:
            path = _safe_artifact_path(root, relative)
        except (FileNotFoundError, ManifestVerificationError) as error:
            raise ManifestVerificationError(f"missing or unsafe artifact: {relative}") from error
        if not path.is_file():
            raise ManifestVerificationError(f"missing or unsafe artifact: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def build_frozen_manifest(
    root: Path,
    relative_paths: Iterable[str],
    *,
    source_commit: str,
    preregistration_ready: bool,
    r14_freeze_pass: bool,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    files = artifact_inventory(root, relative_paths)
    payload = {
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "READY_FOR_SEPARATE_RUN_AUTHORIZATION": bool(preregistration_ready),
        "artifact_count": len(files),
        "artifacts": files,
        "dataset_execution_count": 0,
        "manifest_kind": "SUCCESS_FREEZE" if r14_freeze_pass else "FAILURE_AUDIT_FREEZE",
        "preregistration_ready": bool(preregistration_ready),
        "r14_freeze_pass": bool(r14_freeze_pass),
        "registration_execution_count": 0,
        "source_commit": source_commit,
        "bindings": dict(bindings or {}),
    }
    if preregistration_ready and not r14_freeze_pass:
        raise ManifestVerificationError("ready state requires R14 freeze pass")
    return {**payload, "manifest_root_sha256": compact_sha256(payload)}


def verify_snapshot_selection(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(SNAPSHOT_SELECTION_FIELDS):
            raise ManifestVerificationError("snapshot selection header differs from protocol template")
        rows = list(reader)
    if not rows:
        return {
            "dataset_count": 0,
            "snapshot_count": 0,
            "weak_snapshot_count": 0,
            "rich_snapshot_count": 0,
            "complete_200_inventory": False,
        }
    ids = [row["snapshot_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ManifestVerificationError("duplicate snapshot IDs")
    if any(row["eligibility_status"] != "ELIGIBLE" for row in rows):
        raise ManifestVerificationError("selected inventory contains an ineligible row")
    if any(row["labeler_blinded_to_registration_error"] != "true" for row in rows):
        raise ManifestVerificationError("a selected label is not registration-blind")
    if any(row["open3d_pcl_shared_input"] != "true" for row in rows):
        raise ManifestVerificationError("a selected input is not shared by both backends")
    weak = sum(row["scene_label"] == "CORRIDOR_OR_WEAK_GEOMETRY" for row in rows)
    rich = sum(row["scene_label"] == "GEOMETRY_RICH" for row in rows)
    datasets = sorted({row["dataset_id"] for row in rows})
    counts_ok = all(
        sum(row["dataset_id"] == dataset and row["scene_label"] == label for row in rows) == 50
        for dataset in datasets
        for label in ("CORRIDOR_OR_WEAK_GEOMETRY", "GEOMETRY_RICH")
    )
    return {
        "dataset_count": len(datasets),
        "snapshot_count": len(rows),
        "weak_snapshot_count": weak,
        "rich_snapshot_count": rich,
        "complete_200_inventory": len(datasets) == 2 and len(rows) == 200 and counts_ok,
    }


def _verify_success_only_artifacts(root: Path, selection: Mapping[str, Any]) -> None:
    interval = json.loads((root / "interval_selection_manifest_v1.json").read_text(encoding="utf-8"))
    if interval.get("r14_freeze_pass") is not True or interval.get("status") != "PASS":
        raise ManifestVerificationError("success freeze lacks a passing immutable interval manifest")
    for field, relative in (
        ("canonical_input_manifest_sha256", "canonical_input_manifest.csv"),
        ("snapshot_selection_csv_sha256", "real_data_snapshot_selection_v1.csv"),
    ):
        if interval.get(field) != sha256_file(root / relative):
            raise ManifestVerificationError(f"R14 binding mismatch: {field}")

    with (root / "canonical_input_manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        canonical_rows = list(csv.DictReader(stream))
    if len(canonical_rows) != selection["snapshot_count"]:
        raise ManifestVerificationError("canonical input inventory does not match snapshot inventory")
    canonical_ids = [row.get("snapshot_id") for row in canonical_rows]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ManifestVerificationError("duplicate canonical input snapshot IDs")
    for row in canonical_rows:
        open3d_sha = row.get("future_open3d_bundle_sha256")
        pcl_sha = row.get("future_pcl_bundle_sha256")
        bundle_sha = row.get("bundle_sha256")
        if (
            open3d_sha != pcl_sha
            or open3d_sha != bundle_sha
            or row.get("byte_identical_for_both_backends") != "true"
        ):
            raise ManifestVerificationError("Open3D/PCL canonical bundle binding differs")

    for dataset in ("iilabs", "grandtour"):
        lineage = json.loads((root / dataset / "map_lineage_manifest.json").read_text(encoding="utf-8"))
        map_ids = set(lineage.get("map_scan_source_ids", []))
        query_ids = set(lineage.get("query_scan_source_ids", []))
        if map_ids & query_ids or lineage.get("query_map_intersection_count") != 0:
            raise ManifestVerificationError(f"query/map lineage contamination: {dataset}")

    audit = json.loads((root / "r01_r10_eligibility_audit.json").read_text(encoding="utf-8"))
    requirements = audit.get("requirements", [])
    if len(requirements) != 10 or any(row.get("global_status") != "PASS" for row in requirements):
        raise ManifestVerificationError("success freeze does not have ten passing eligibility gates")


def verify_frozen_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    payload = {key: value for key, value in manifest.items() if key != "manifest_root_sha256"}
    if compact_sha256(payload) != manifest.get("manifest_root_sha256"):
        raise ManifestVerificationError("manifest root hash mismatch")
    recorded_paths: set[str] = set()
    for row in manifest.get("artifacts", []):
        relative = row["path"]
        if relative in recorded_paths:
            raise ManifestVerificationError(f"duplicate manifest path: {relative}")
        recorded_paths.add(relative)
        try:
            path = _safe_artifact_path(root, relative)
        except FileNotFoundError as error:
            raise ManifestVerificationError(f"missing manifest artifact: {relative}") from error
        if not path.is_file():
            raise ManifestVerificationError(f"missing manifest artifact: {relative}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ManifestVerificationError(f"artifact checksum mismatch: {relative}")
    expected_files = recorded_paths | {"frozen_manifest_v1.json", "SHA256SUMS"}
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    orphan_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    # In-memory unit construction intentionally precedes materializing the two
    # envelope files; an on-disk formal manifest must be strictly closed.
    if (root / "frozen_manifest_v1.json").exists() and (orphan_files or missing_files):
        raise ManifestVerificationError(
            f"manifest closure mismatch: missing={missing_files}, orphan={orphan_files}"
        )
    checksum_path = root / "SHA256SUMS"
    if checksum_path.exists():
        checksum_paths: set[str] = set()
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            try:
                digest, relative = line.split("  ", 1)
                path = _safe_artifact_path(root, relative)
            except (ValueError, FileNotFoundError, ManifestVerificationError) as error:
                raise ManifestVerificationError(f"invalid SHA256SUMS line: {line}") from error
            if relative in checksum_paths or relative == "SHA256SUMS" or sha256_file(path) != digest:
                raise ManifestVerificationError(f"SHA256SUMS mismatch: {relative}")
            checksum_paths.add(relative)
        if checksum_paths != recorded_paths | {"frozen_manifest_v1.json"}:
            raise ManifestVerificationError("SHA256SUMS inventory differs from frozen manifest")

    bindings = manifest.get("bindings", {})
    expected_bindings = {
        "backend_parameter_contract_file_sha256": BACKEND_FILE_SHA256,
        "open3d_parameter_canonical_sha256": OPEN3D_PARAMETERS_SHA256,
        "pcl_parameter_canonical_sha256": PCL_PARAMETERS_SHA256,
    }
    if any(bindings.get(key) != value for key, value in expected_bindings.items()):
        raise ManifestVerificationError("backend parameter binding mismatch")
    if repository_root is not None:
        repository = repository_root.resolve(strict=True)
        if sha256_file(repository / "frozen_assets/backend_parameter_contract.json") != BACKEND_FILE_SHA256:
            raise ManifestVerificationError("live backend parameter contract differs from frozen SHA")

    for key in (
        "REAL_DATA_RUN_AUTHORIZED",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
    ):
        if manifest.get(key) is not False:
            raise ManifestVerificationError(f"unauthorized state in preregistration manifest: {key}")
    if manifest.get("dataset_execution_count") != 0 or manifest.get("registration_execution_count") != 0:
        raise ManifestVerificationError("execution count is nonzero in preregistration manifest")
    selection = verify_snapshot_selection(root / "real_data_snapshot_selection_v1.csv")
    attestation = json.loads((root / "NO_ICP_ATTESTATION.json").read_text(encoding="utf-8"))
    zero_counts = attestation.get("pass") is True and all(
        attestation.get(key) == 0
        for key in (
            "open3d_registration_call_count",
            "pcl_cli_invocation_count",
            "other_registration_process_count",
            "real_trial_result_count",
            "estimated_transform_file_count",
            "registration_execution_count",
        )
    )
    success = (
        manifest.get("manifest_kind") == "SUCCESS_FREEZE"
        and manifest.get("r14_freeze_pass") is True
        and selection["complete_200_inventory"] is True
        and zero_counts
    )
    if success:
        _verify_success_only_artifacts(root, selection)
    return {
        "artifact_integrity_pass": True,
        "missing_file_count": len(missing_files),
        "no_registration_pass": zero_counts,
        "orphan_file_count": len(orphan_files),
        "preregistration_verification_pass": success,
        "selection": selection,
    }


def sha256sums_bytes(root: Path, relative_paths: Sequence[str]) -> bytes:
    lines = [f"{sha256_file(root / relative)}  {relative}" for relative in sorted(relative_paths)]
    return ("\n".join(lines) + "\n").encode("utf-8")
