"""Independent, fail-closed verifier for the CAVERS Stage-1 audit.

The verifier deliberately does not import :mod:`cavers_stage1`.  It rebuilds
the relevant conclusions from the pinned official metadata and the
materialized GT/TF/timestamp evidence.  This is important for the expected
fail-closed result: a shared string named ``map`` is not treated as proof that
twelve recordings share one calibrated OptiTrack world.
"""

from __future__ import annotations

import binascii
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


ZENODO_RECORD_ID = 19367714
ZENODO_CONCEPT_RECORD_ID = "19367713"
ZENODO_DOI = "10.5281/zenodo.19367714"
ZENODO_VERSION = "0.0.1"
ZENODO_REVISION = 18
ZENODO_LICENSE = "cc-by-4.0"
ZENODO_PUBLICATION_DATE = "2026-04-01"
ZENODO_TOTAL_ARCHIVE_BYTES = 162_022_890_726
ZENODO_METADATA_SHA256 = "5047d09c5c7e39f0df861b1773edf1304aa1d0448bbc3a82b7396aae147f0c24"
GITHUB_COMMIT = "74ead2bf1cffa337a1bff1b03b990819cb7ff067"
GITHUB_REPOSITORY_URL = "https://github.com/spaceuma/cavers.git"
GITHUB_CODE_LICENSE = "MIT"
ARXIV_PDF_SHA256 = "ea7cfc4385c768b3144958952d0923ccd09e88f4b56c3814544328b698c98029"
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
MAX_STAGE1_FILE_BYTES = 500_000_000

CANDIDATE_SEQUENCES = tuple(
    [f"loc_diablo_{index}" for index in range(1, 9)]
    + [f"loc_handheld_{index}" for index in range(1, 5)]
)
EXCLUDED_NO_GT_SEQUENCES = ("loc_handheld_5", "loc_handheld_6")

OVERLAP_CONTRACT = {
    "maximum_native_gap_s": 0.2,
    "min_contiguous_covered_duration_s": 5.0,
    "min_coverage_fraction": 0.60,
    "min_eligible_nonoverlapping_5s_intervals": 30,
    "min_total_covered_duration_s": 150.0,
    "radius_m": 5.0,
    "resample_rate_hz": 1.0,
}
PAIR_RANKING_RULE = [
    "covered duration descending",
    "coverage fraction descending",
    "eligible 5s intervals descending",
    "nearest distance q95 ascending",
    "map ID lexicographic",
    "query ID lexicographic",
]

REQUIRED_RUNTIME_FILES = frozenset(
    {
        "cleanup_report.json",
        "cavers_cleanup_report.json",
        "environment_report.json",
        "official_source_manifest.json",
        "zenodo_record_metadata.json",
        "zenodo_file_inventory.csv",
        "github_source_manifest.json",
        "download_manifest.json",
        "download_manifest.csv",
        "license_and_citation_report.md",
        "cavers_sequence_inventory.csv",
        "cavers_sequence_inventory.json",
        "cavers_reference_audit.csv",
        "cavers_reference_audit.json",
        "cavers_common_world_frame_audit.json",
        "cavers_extrinsic_audit.json",
        "cavers_transform_chain_manifest.json",
        "cavers_time_sync_audit.json",
        "cavers_stage1_uncertainty_feasibility.csv",
        "cavers_stage1_uncertainty_feasibility.json",
        "cavers_gt_only_overlap_matrix.csv",
        "cavers_gt_only_overlap_matrix.json",
        "cavers_pair_selection.json",
        "cavers_stage1_eligibility.json",
        "NO_ICP_ATTESTATION.json",
        "cavers_stage1_summary.md",
        "cavers_stage1_summary.json",
        "cavers_stage1_manifest.json",
        "SHA256SUMS",
    }
)
RESUME_ENVIRONMENT_FILE = "resume_environment_report.json"
RESUME_ENVIRONMENT_EXTRA_PATTERN = re.compile(
    r"resume_environment_report_([0-9a-f]{40})\.json"
)
RESUME_POLICY = (
    "authenticate completed sequence checkpoints by official archive identity, "
    "size, CRC32, and SHA-256 before continuation"
)

GT_REQUIRED_FIELDS = (
    "Timestamp",
    "Frame_ID",
    "Child_Frame_ID",
    "PX",
    "PY",
    "PZ",
    "QX",
    "QY",
    "QZ",
    "QW",
)
GT_VELOCITY_FIELDS = ("VX", "VY", "VZ", "VROLL", "VPITCH", "VYAW")
REFERENCE_AUDIT_FIELDS = (
    "sequence_id",
    "status",
    "complete_6dof",
    "independent_reference_source",
    "world_frames",
    "child_frames",
    "row_count",
    "first_timestamp_s",
    "last_timestamp_s",
    "duration_s",
    "median_rate_hz",
    "maximum_timestamp_gap_s",
    "strictly_monotonic",
    "position_finite",
    "first_position_m",
    "first_quaternion_xyzw",
    "timestamp_scale_to_seconds",
    "quaternion_order",
    "quaternion_norm_pass",
    "max_quaternion_norm_error",
    "linear_and_angular_velocity_fields_present",
    "rosbag_optitrack_topic_present",
    "rosbag_optitrack_message_count",
    "sequence_local_reset_detected",
    "gt_sha256",
)
UNCERTAINTY_COMPONENTS = {
    "OptiTrack position uncertainty": "m",
    "OptiTrack orientation uncertainty": "rad",
    "GT timestamp uncertainty": "s",
    "LiDAR timestamp uncertainty": "s",
    "synchronization/delay uncertainty": "s",
    "CAD extrinsic translation uncertainty": "m",
    "CAD extrinsic rotation uncertainty": "rad",
    "interpolation uncertainty": "m/rad",
    "future map accumulation uncertainty": "m",
    "motion distortion / deskew uncertainty": "m",
}

ZERO_REGISTRATION_FIELDS = frozenset(
    {
        "actual_registration_execution_count",
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
    }
)
SUSPICIOUS_RESULT_FIELDS = frozenset(
    {
        "t_estimated",
        "estimated_transform",
        "estimated_transform_4x4",
        "estimated_pose",
        "estimated_pose_4x4",
        "estimated_trajectory",
        "final_transform",
        "final_transform_4x4",
        "registration_result",
        "registration_results",
        "registration_error",
        "registration_error_m",
        "registration_error_rad",
        "registration_residual",
        "registration_residual_rmse",
        "final_residual",
        "final_residual_rmse",
        "icp_result",
        "gicp_result",
        "ndt_result",
        "scan_matching_result",
    }
)
SUSPICIOUS_FILENAME_TOKENS = (
    "t_estimated",
    "estimated_transform",
    "estimated_pose",
    "estimated_trajectory",
    "final_transform",
    "registration_result",
    "registration_output",
    "registration_error",
    "registration_residual",
    "icp_result",
    "gicp_result",
    "ndt_result",
    "scan_matching_result",
)
FORBIDDEN_DATA_SUFFIXES = frozenset({".bag", ".db3", ".las", ".laz", ".mcap", ".pcd", ".ply"})


class CaversStage1VerificationError(RuntimeError):
    """The CAVERS Stage-1 artifact or its scientific conclusion is invalid."""


def _fail(message: str) -> None:
    raise CaversStage1VerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except CaversStage1VerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CaversStage1VerificationError(f"invalid JSON: {path}: {error}") from error


def _mapping_json(path: Path) -> dict[str, Any]:
    value = _json(path)
    _require(isinstance(value, dict), f"JSON root is not an object: {path.name}")
    return value


def _csv(path: Path, expected_fields: Sequence[str] | None = None) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            if expected_fields is not None:
                _require(fields == tuple(expected_fields), f"CSV header mismatch: {path.name}")
            rows = list(reader)
    except CaversStage1VerificationError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise CaversStage1VerificationError(f"invalid CSV: {path}: {error}") from error
    _require(all(None not in row for row in rows), f"CSV has excess columns: {path.name}")
    return rows


def _number(value: Any, description: str) -> float:
    _require(not isinstance(value, bool), f"{description} is boolean, not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CaversStage1VerificationError(f"invalid number for {description}: {value!r}") from error
    _require(math.isfinite(result), f"non-finite number for {description}")
    return result


def _same_number(actual: Any, expected: Any, description: str, *, atol: float = 1e-9) -> None:
    left = _number(actual, description)
    right = _number(expected, description)
    _require(abs(left - right) <= atol, f"numeric mismatch for {description}: {left} != {right}")


def _same_vector(actual: Any, expected: Sequence[float], description: str, *, atol: float = 1e-12) -> None:
    _require(isinstance(actual, list) and len(actual) == len(expected), f"invalid vector: {description}")
    for index, expected_value in enumerate(expected):
        _same_number(actual[index], expected_value, f"{description}[{index}]", atol=atol)


def _safe_relative(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    _require(
        not posix.is_absolute() and bool(posix.parts) and "." not in posix.parts and ".." not in posix.parts,
        f"unsafe relative path: {relative}",
    )
    path = root.joinpath(*posix.parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CaversStage1VerificationError(f"missing path: {relative}") from error
    _require(resolved == path, f"symlink or non-canonical artifact: {relative}")
    _require(resolved == root or root in resolved.parents, f"path escapes root: {relative}")
    _require(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {relative}")
    return path


def _runtime_file_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail(f"runtime symlink is forbidden: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _require("/" not in relative, f"runtime payload must be flat: {relative}")
            names.add(relative)
    return names


def _verify_sha256sums(root: Path, actual_files: set[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    _require(bool(lines), "SHA256SUMS is empty")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        _require(match is not None, f"malformed SHA256SUMS line: {line!r}")
        assert match is not None
        digest, relative = match.groups()
        _require(relative not in entries and relative != "SHA256SUMS", f"duplicate/recursive SHA entry: {relative}")
        path = _safe_relative(root, relative)
        _require(_sha256(path) == digest, f"SHA256SUMS mismatch: {relative}")
        entries[relative] = digest
    _require(set(entries) == actual_files - {"SHA256SUMS"}, "SHA256SUMS inventory is not closed")
    return entries


def _git_commit_is_head_ancestor(repository: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _git_commit_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _resume_environment_paths(root: Path) -> list[Path]:
    standard = root / RESUME_ENVIRONMENT_FILE
    extras = sorted(root.glob("resume_environment_report_*.json"))
    for path in extras:
        _require(
            RESUME_ENVIRONMENT_EXTRA_PATTERN.fullmatch(path.name) is not None,
            f"malformed immutable resume environment filename: {path.name}",
        )
    if extras:
        _require(standard.is_file(), "additional resume provenance exists without the standard first report")
    return ([standard] if standard.is_file() else []) + extras


def _verify_resume_provenance(
    *,
    root: Path,
    repository: Path,
    producer_commit: str,
    finalizing_commit: str,
) -> list[str]:
    paths = _resume_environment_paths(root)
    if not paths:
        _require(finalizing_commit == producer_commit, "fresh run finalizing commit differs from producer commit")
        return []

    reports_by_commit: dict[str, dict[str, Any]] = {}
    standard_commit: str | None = None
    for index, path in enumerate(paths):
        environment = _mapping_json(path)
        git_value = environment.get("git")
        _require(
            isinstance(git_value, dict)
            and isinstance(git_value.get("commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", git_value["commit"]) is not None
            and git_value.get("branch") == "prep/cavers-single-dataset-stage1-v1"
            and git_value.get("worktree_porcelain") == [],
            f"resume environment Git provenance mismatch: {path.name}",
        )
        commit = git_value["commit"]
        if index > 0:
            match = RESUME_ENVIRONMENT_EXTRA_PATTERN.fullmatch(path.name)
            assert match is not None
            _require(match.group(1) == commit, f"resume environment filename/commit mismatch: {path.name}")
            previous = environment.get("previous_resume_commit")
            _require(
                isinstance(previous, str)
                and re.fullmatch(r"[0-9a-f]{40}", previous) is not None,
                f"additional resume report lacks a valid previous commit: {path.name}",
            )
        else:
            _require(
                "previous_resume_commit" not in environment,
                "standard first resume report unexpectedly has a previous commit",
            )
            standard_commit = commit
        _require(
            environment.get("resume_origin_commit") == producer_commit
            and environment.get("resume_policy") == RESUME_POLICY,
            f"resume origin/policy provenance mismatch: {path.name}",
        )
        _require(
            _git_commit_is_head_ancestor(repository, commit),
            f"resume provenance commit is not an ancestor of current HEAD: {commit}",
        )
        _require(commit not in reports_by_commit, "duplicate commit in immutable resume provenance chain")
        reports_by_commit[commit] = environment

    assert standard_commit is not None
    _require(
        _git_commit_is_ancestor(repository, producer_commit, standard_commit),
        "standard resume provenance commit does not descend from the producer commit",
    )
    chain = [standard_commit]
    consumed = {standard_commit}
    latest_commit = standard_commit
    while True:
        successors = [
            commit
            for commit, report in reports_by_commit.items()
            if commit not in consumed and report.get("previous_resume_commit") == latest_commit
        ]
        if not successors:
            break
        _require(len(successors) == 1, "resume provenance chain forks")
        successor = successors[0]
        _require(
            _git_commit_is_ancestor(repository, latest_commit, successor),
            "resume provenance pointer is not forward-only in Git ancestry",
        )
        chain.append(successor)
        consumed.add(successor)
        latest_commit = successor
    _require(consumed == set(reports_by_commit), "resume provenance chain is disconnected")
    _require(finalizing_commit == chain[-1], "manifest finalizing commit is not the resume chain tail")
    return chain


def _verify_manifest(
    root: Path,
    data_rows: list[dict[str, Any]],
    repository: Path,
) -> dict[str, Any]:
    manifest = _mapping_json(root / "cavers_stage1_manifest.json")
    stored = manifest.get("manifest_payload_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    _require(stored == _compact_sha256(payload), "CAVERS manifest payload SHA mismatch")
    _require(manifest.get("schema_version") == "cavers_stage1_manifest_v1", "manifest schema mismatch")
    identity = manifest.get("official_source_identity")
    _require(
        identity
        == {
            "github_commit": GITHUB_COMMIT,
            "zenodo_record_id": ZENODO_RECORD_ID,
            "zenodo_revision": ZENODO_REVISION,
            "zenodo_version": ZENODO_VERSION,
        },
        "manifest official source identity mismatch",
    )
    _require(manifest.get("data_evidence") == data_rows, "manifest data evidence differs from live small files")
    _require(
        manifest.get("eligibility_sha256") == _sha256(root / "cavers_stage1_eligibility.json"),
        "manifest eligibility SHA mismatch",
    )
    producer_commit = manifest.get("producer_commit")
    _require(isinstance(producer_commit, str) and re.fullmatch(r"[0-9a-f]{40}", producer_commit) is not None, "invalid producer commit")
    finalizing_commit = manifest.get("finalizing_commit")
    _require(
        isinstance(finalizing_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", finalizing_commit) is not None,
        "invalid finalizing commit",
    )
    environment = _mapping_json(root / "environment_report.json")
    environment_git = environment.get("git")
    _require(
        isinstance(environment_git, dict)
        and environment_git.get("commit") == producer_commit
        and environment_git.get("branch") == "prep/cavers-single-dataset-stage1-v1"
        and environment_git.get("worktree_porcelain") == [],
        "fresh producer environment provenance mismatch",
    )
    _verify_resume_provenance(
        root=root,
        repository=repository,
        producer_commit=producer_commit,
        finalizing_commit=finalizing_commit,
    )
    _require(
        _git_commit_is_head_ancestor(repository, producer_commit),
        "producer commit is not an ancestor of current HEAD",
    )
    _require(
        _git_commit_is_head_ancestor(repository, finalizing_commit),
        "finalizing commit is not an ancestor of current HEAD",
    )

    rows = manifest.get("payload")
    _require(isinstance(rows, list), "manifest payload is not a list")
    expected_paths = _runtime_file_names(root) - {"SHA256SUMS", "cavers_stage1_manifest.json"}
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), "invalid manifest payload row")
        relative = row.get("path")
        _require(isinstance(relative, str) and relative not in seen, "duplicate/invalid manifest payload path")
        seen.add(relative)
        path = _safe_relative(root, relative)
        _require(row.get("size_bytes") == path.stat().st_size, f"manifest size mismatch: {relative}")
        _require(row.get("sha256") == _sha256(path), f"manifest SHA mismatch: {relative}")
    _require(seen == expected_paths, "manifest payload inventory is not closed")
    return manifest


def _scan_data_files(data_root: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    maximum_size = 0
    for path in sorted(data_root.rglob("*")):
        if path.is_symlink():
            _fail(f"data symlink is forbidden: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        maximum_size = max(maximum_size, size)
        _require(size <= MAX_STAGE1_FILE_BYTES, f">500 MB file materialized: {path}")
        if ".git" not in path.parts:
            _require(path.suffix.lower() not in FORBIDDEN_DATA_SUFFIXES, f"forbidden point-cloud/rosbag payload: {path}")
            rows.append(
                {
                    "local_path": str(path),
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": size,
                    "status": "MATERIALIZED_SMALL_STAGE1_EVIDENCE",
                }
            )
    _require(bool(rows), "no Stage-1 source evidence exists")
    return rows, maximum_size


def _verify_download_manifest(runtime_root: Path, data_rows: list[dict[str, Any]], maximum_size: int) -> dict[str, Any]:
    value = _mapping_json(runtime_root / "download_manifest.json")
    _require(value.get("materialized_files") == data_rows, "download manifest rows differ from live files")
    _require(value.get("materialized_bytes") == sum(row["size_bytes"] for row in data_rows), "materialized byte total mismatch")
    _require(value.get("maximum_materialized_file_bytes") == max(row["size_bytes"] for row in data_rows), "maximum materialized file mismatch")
    _require(maximum_size <= MAX_STAGE1_FILE_BYTES, "live data maximum exceeds Stage-1 limit")
    maximum_download = value.get("maximum_single_download_bytes")
    _require(isinstance(maximum_download, int) and 0 < maximum_download <= MAX_STAGE1_FILE_BYTES, "invalid maximum download limit")
    for key in ("full_archive_download_count", "large_lidar_or_point_cloud_download_count"):
        _require(value.get(key) == 0 and not isinstance(value.get(key), bool), f"nonzero {key}")

    fields = ("relative_path", "local_path", "size_bytes", "sha256", "status")
    csv_rows = _csv(runtime_root / "download_manifest.csv", fields)
    expected_csv = [
        {
            "relative_path": row["relative_path"],
            "local_path": row["local_path"],
            "size_bytes": str(row["size_bytes"]),
            "sha256": row["sha256"],
            "status": row["status"],
        }
        for row in data_rows
    ]
    _require(csv_rows == expected_csv, "download CSV differs from live small-file inventory")
    return value


def _zenodo_file_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = raw.get("files")
    _require(isinstance(files, list) and len(files) == 48, "Zenodo record must contain 48 archives")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in files:
        _require(isinstance(source, dict), "invalid Zenodo file row")
        name = source.get("key")
        size = source.get("size")
        checksum = source.get("checksum")
        links = source.get("links")
        url = links.get("self") if isinstance(links, dict) else None
        _require(isinstance(name, str) and name not in seen, "duplicate/invalid Zenodo archive name")
        _require(isinstance(size, int) and size > MAX_STAGE1_FILE_BYTES, f"unexpected Zenodo archive size: {name}")
        _require(isinstance(checksum, str) and re.fullmatch(r"md5:[0-9a-f]{32}", checksum) is not None, f"invalid Zenodo checksum: {name}")
        _require(isinstance(url, str) and url.startswith("https://zenodo.org/api/records/19367714/files/"), f"non-official Zenodo URL: {name}")
        seen.add(name)
        rows.append({"archive_name": name, "checksum": checksum, "download_url": url, "size_bytes": size})
    return sorted(rows, key=lambda row: row["archive_name"])


def _verify_official_source(runtime_root: Path, data_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    runtime_raw = runtime_root / "zenodo_record_metadata.json"
    source_raw = data_root / "source_metadata" / "zenodo_record_metadata.json"
    _require(runtime_raw.read_bytes() == source_raw.read_bytes(), "runtime/source Zenodo metadata bytes differ")
    _require(_sha256(runtime_raw) == ZENODO_METADATA_SHA256, "exact Zenodo metadata SHA mismatch")
    raw = _mapping_json(runtime_raw)
    metadata = raw.get("metadata")
    _require(isinstance(metadata, dict), "Zenodo metadata object missing")
    license_value = metadata.get("license")
    _require(
        raw.get("id") == ZENODO_RECORD_ID
        and str(raw.get("conceptrecid")) == ZENODO_CONCEPT_RECORD_ID
        and raw.get("doi") == ZENODO_DOI
        and raw.get("revision") == ZENODO_REVISION
        and metadata.get("version") == ZENODO_VERSION
        and metadata.get("publication_date") == ZENODO_PUBLICATION_DATE
        and raw.get("status") == "published"
        and raw.get("state") == "done"
        and raw.get("submitted") is True
        and isinstance(license_value, dict)
        and license_value.get("id") == ZENODO_LICENSE,
        "official Zenodo revision/version/DOI/license mismatch",
    )
    zenodo_rows = _zenodo_file_rows(raw)
    _require(sum(row["size_bytes"] for row in zenodo_rows) == ZENODO_TOTAL_ARCHIVE_BYTES, "Zenodo archive byte total mismatch")
    inventory = _csv(
        runtime_root / "zenodo_file_inventory.csv",
        ("archive_name", "size_bytes", "checksum", "download_url"),
    )
    expected_inventory = [
        {
            "archive_name": row["archive_name"],
            "size_bytes": str(row["size_bytes"]),
            "checksum": row["checksum"],
            "download_url": row["download_url"],
        }
        for row in zenodo_rows
    ]
    _require(inventory == expected_inventory, "Zenodo CSV inventory differs from exact record")

    github_root = data_root / "source_metadata" / "github_repository"
    github = _mapping_json(runtime_root / "github_source_manifest.json")
    _require(
        github.get("commit") == GITHUB_COMMIT
        and github.get("repository_url") == GITHUB_REPOSITORY_URL
        and github.get("code_license") == GITHUB_CODE_LICENSE,
        "official GitHub identity/license mismatch",
    )
    try:
        live_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=github_root, text=True, stderr=subprocess.STDOUT
        ).strip()
        live_status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=github_root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CaversStage1VerificationError(f"cannot verify pinned GitHub checkout: {error}") from error
    _require(live_commit == GITHUB_COMMIT and not live_status, "GitHub checkout is not the pinned clean commit")
    github_rows = [
        {
            "path": path.relative_to(github_root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(github_root.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    ]
    _require(github.get("files") == github_rows, "GitHub small-file inventory mismatch")
    _require(github.get("tree_rows_sha256") == _compact_sha256(github_rows), "GitHub tree digest mismatch")
    _require((github_root / "LICENSE").is_file() and "MIT License" in (github_root / "LICENSE").read_text(encoding="utf-8"), "MIT license text missing")

    official = _mapping_json(runtime_root / "official_source_manifest.json")
    paper_path = data_root / "source_metadata" / "cavers_paper_2604.15052.pdf"
    _require(_sha256(paper_path) == ARXIV_PDF_SHA256, "exact official arXiv PDF SHA mismatch")
    expected_record = {
        "created": raw.get("created"),
        "doi": ZENODO_DOI,
        "file_count": 48,
        "license": ZENODO_LICENSE,
        "modified": raw.get("modified"),
        "publication_date": ZENODO_PUBLICATION_DATE,
        "record_id": ZENODO_RECORD_ID,
        "revision": ZENODO_REVISION,
        "title": metadata.get("title"),
        "total_archive_bytes": sum(row["size_bytes"] for row in zenodo_rows),
        "version": ZENODO_VERSION,
    }
    _require(
        official.get("arxiv_id") == "2604.15052"
        and official.get("dataset_license") == ZENODO_LICENSE
        and official.get("github_code_license") == GITHUB_CODE_LICENSE
        and official.get("github_commit") == GITHUB_COMMIT
        and official.get("github_url") == GITHUB_REPOSITORY_URL
        and official.get("large_archive_download_count") == 0
        and official.get("zenodo_metadata_sha256") == ZENODO_METADATA_SHA256
        and official.get("arxiv_pdf")
        == {
            "path": str(paper_path),
            "sha256": ARXIV_PDF_SHA256,
            "url": "https://arxiv.org/pdf/2604.15052",
        }
        and official.get("record") == expected_record,
        "official source manifest mismatch",
    )
    static = official.get("static_no_registration_audit")
    _require(isinstance(static, dict) and static.get("pass") is True and static.get("violations") == [], "static no-registration audit failed")
    return official, {row["archive_name"]: row for row in zenodo_rows}


def _verify_small_members(
    official: Mapping[str, Any],
    archive_rows: Mapping[str, Mapping[str, Any]],
    data_root: Path,
    runtime_root: Path,
) -> None:
    evidence = official.get("source_evidence")
    _require(isinstance(evidence, dict), "official source evidence missing")
    sequences = evidence.get("sequences")
    _require(isinstance(sequences, list) and [row.get("sequence_id") for row in sequences if isinstance(row, dict)] == list(CANDIDATE_SEQUENCES), "source sequence evidence mismatch")
    source_root = data_root / "source_metadata"
    receipt_path = source_root / "http_range_receipts.json"
    _require(evidence.get("range_receipt_path") == str(receipt_path), "range receipt path mismatch")
    _require(evidence.get("range_receipt_sha256") == _sha256(receipt_path), "range receipt SHA mismatch")
    receipts = _mapping_json(receipt_path)
    _require(receipts.get("full_archive_download_count") == 0, "full archive download was recorded")
    maximum = receipts.get("maximum_materialized_member_bytes")
    _require(isinstance(maximum, int) and 0 < maximum <= MAX_STAGE1_FILE_BYTES, "invalid member byte guard")
    receipt_rows = receipts.get("receipts")
    _require(isinstance(receipt_rows, list) and bool(receipt_rows), "HTTP range receipts missing")
    response_total = 0
    for row in receipt_rows:
        _require(isinstance(row, dict), "invalid HTTP receipt")
        response_bytes = row.get("response_bytes")
        _require(isinstance(response_bytes, int) and 0 < response_bytes <= maximum, "HTTP receipt exceeds byte guard")
        _require(row.get("status") in (200, 206), "unexpected HTTP receipt status")
        _require(isinstance(row.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid HTTP receipt SHA")
        _require(str(row.get("url", "")).startswith("https://zenodo.org/api/records/19367714/files/"), "non-official HTTP receipt URL")
        response_total += response_bytes
    _require(receipts.get("total_range_response_bytes") == response_total, "HTTP receipt total mismatch")
    _require(evidence.get("total_range_response_bytes") == response_total, "official receipt total mismatch")
    resume_ids = receipts.get("resume_authenticated_sequence_ids")
    resume_count = receipts.get("resume_authenticated_sequence_count")
    _require(isinstance(resume_ids, list), "resume-authenticated sequence IDs missing")
    _require(
        all(isinstance(sequence_id, str) and sequence_id in CANDIDATE_SEQUENCES for sequence_id in resume_ids),
        "resume-authenticated sequence ID is not a CAVERS candidate",
    )
    _require(len(resume_ids) == len(set(resume_ids)), "duplicate resume-authenticated sequence ID")
    _require(
        isinstance(resume_count, int)
        and not isinstance(resume_count, bool)
        and resume_count == len(resume_ids),
        "resume-authenticated sequence count mismatch",
    )
    if not (runtime_root / RESUME_ENVIRONMENT_FILE).is_file():
        _require(resume_count == 0, "checkpoint reuse recorded without resume environment provenance")

    expected_suffixes = {
        "GT_ODOM/data.csv",
        "TF/data.csv",
        "TF_STATIC/data.csv",
        "VELODYNE_CLOUD/data.csv",
        "rosbag_metadata.yaml",
    }
    for sequence_row in sequences:
        _require(isinstance(sequence_row, dict), "invalid sequence evidence row")
        sequence_id = sequence_row["sequence_id"]
        inventory_path = source_root / "sequences" / sequence_id / "remote_zip_inventory.json"
        _require(sequence_row.get("zip_inventory_path") == str(inventory_path), f"ZIP inventory path mismatch: {sequence_id}")
        _require(sequence_row.get("zip_inventory_sha256") == _sha256(inventory_path), f"ZIP inventory SHA mismatch: {sequence_id}")
        inventory = _mapping_json(inventory_path)
        _require(inventory.get("sequence_id") == sequence_id, f"ZIP sequence mismatch: {sequence_id}")
        raw_name = f"{sequence_id}.zip"
        bag_name = f"{sequence_id}_rosbag.zip"
        _require(raw_name in archive_rows and bag_name in archive_rows, f"official archives missing: {sequence_id}")
        for key, name in (("raw_archive", raw_name), ("rosbag_archive", bag_name)):
            archive = inventory.get(key)
            _require(
                isinstance(archive, dict)
                and archive.get("archive_name") == name
                and archive.get("archive_size_bytes") == archive_rows[name]["size_bytes"]
                and archive.get("url") == archive_rows[name]["download_url"],
                f"remote ZIP identity mismatch: {name}",
            )
        members = inventory.get("selected_members")
        _require(isinstance(members, list) and len(members) == 5, f"selected member count mismatch: {sequence_id}")
        actual_suffixes: set[str] = set()
        evidence_bytes = 0
        for member in members:
            _require(isinstance(member, dict), f"invalid selected member: {sequence_id}")
            local = Path(str(member.get("local_path")))
            _require(local.is_absolute() and local.resolve(strict=True) == local and source_root in local.parents, f"unsafe member path: {local}")
            _require(local.is_file() and not local.is_symlink(), f"missing selected member: {local}")
            relative = local.relative_to(source_root / "sequences" / sequence_id).as_posix()
            actual_suffixes.add(relative)
            payload = local.read_bytes()
            evidence_bytes += len(payload)
            _require(member.get("uncompressed_size") == len(payload), f"member size mismatch: {local}")
            _require(member.get("sha256") == hashlib.sha256(payload).hexdigest(), f"small-file SHA mismatch: {local}")
            _require(member.get("crc32") == f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}", f"small-file CRC mismatch: {local}")
            expected_archive = bag_name if relative == "rosbag_metadata.yaml" else raw_name
            _require(member.get("archive_name") == expected_archive, f"member archive provenance mismatch: {local}")
        _require(actual_suffixes == expected_suffixes, f"small-file allow-list mismatch: {sequence_id}")
        _require(sequence_row.get("selected_member_count") == 5 and sequence_row.get("evidence_bytes") == evidence_bytes, f"sequence evidence total mismatch: {sequence_id}")
        _require(sequence_row.get("raw_archive_name") == raw_name and sequence_row.get("raw_archive_bytes_not_downloaded") == archive_rows[raw_name]["size_bytes"], f"raw archive non-download record mismatch: {sequence_id}")
        _require(sequence_row.get("rosbag_archive_name") == bag_name and sequence_row.get("rosbag_archive_bytes_not_downloaded") == archive_rows[bag_name]["size_bytes"], f"rosbag non-download record mismatch: {sequence_id}")


def _float_field(row: Mapping[str, str], name: str, sequence_id: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise CaversStage1VerificationError(f"invalid {name} in {sequence_id}") from error
    _require(math.isfinite(value), f"non-finite {name} in {sequence_id}")
    return value


def _audit_gt(path: Path, sequence_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        _require(all(name in fields for name in GT_REQUIRED_FIELDS), f"incomplete GT 6DoF schema: {sequence_id}")
        rows = list(reader)
    _require(len(rows) >= 2, f"GT has fewer than two rows: {sequence_id}")
    raw_times = [_float_field(row, "Timestamp", sequence_id) for row in rows]
    scale = 1e-9 if statistics.median(raw_times) > 1e12 else 1.0
    times = [value * scale for value in raw_times]
    gaps = [right - left for left, right in zip(times, times[1:])]
    positions = [tuple(_float_field(row, name, sequence_id) for name in ("PX", "PY", "PZ")) for row in rows]
    quaternions = [tuple(_float_field(row, name, sequence_id) for name in ("QX", "QY", "QZ", "QW")) for row in rows]
    norms = [math.sqrt(sum(value * value for value in quaternion)) for quaternion in quaternions]
    maximum_norm_error = max(abs(value - 1.0) for value in norms)
    first_q = quaternions[0]
    first_angle = 2.0 * math.atan2(math.sqrt(sum(value * value for value in first_q[:3])), abs(first_q[3]))
    local_reset = math.sqrt(sum(value * value for value in positions[0])) < 1e-6 and first_angle < 1e-6
    world_frames = sorted({row["Frame_ID"] for row in rows})
    child_frames = sorted({row["Child_Frame_ID"] for row in rows})
    monotonic = all(gap > 0.0 for gap in gaps)
    quaternion_pass = maximum_norm_error <= 1e-6
    status = "PASS" if monotonic and quaternion_pass and len(world_frames) == 1 and len(child_frames) == 1 else "FAIL"
    return {
        "child_frames": child_frames,
        "complete_6dof": True,
        "duration_s": times[-1] - times[0],
        "first_position_m": list(positions[0]),
        "first_quaternion_xyzw": list(first_q),
        "first_timestamp_s": times[0],
        "gt_sha256": _sha256(path),
        "independent_reference_source": "OptiTrack /spaceuma/optitrack/odom",
        "last_timestamp_s": times[-1],
        "linear_and_angular_velocity_fields_present": all(name in fields for name in GT_VELOCITY_FIELDS),
        "max_quaternion_norm_error": maximum_norm_error,
        "maximum_timestamp_gap_s": max(gaps),
        "median_rate_hz": 1.0 / statistics.median(gaps),
        "position_finite": True,
        "quaternion_norm_pass": quaternion_pass,
        "quaternion_order": "qx,qy,qz,qw (scalar-last)",
        "row_count": len(rows),
        "sequence_id": sequence_id,
        "sequence_local_reset_detected": local_reset,
        "status": status,
        "strictly_monotonic": monotonic,
        "timestamp_scale_to_seconds": scale,
        "world_frames": world_frames,
    }


def _compare_reference_row(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    exact_keys = (
        "child_frames",
        "complete_6dof",
        "gt_sha256",
        "independent_reference_source",
        "linear_and_angular_velocity_fields_present",
        "position_finite",
        "quaternion_norm_pass",
        "quaternion_order",
        "row_count",
        "sequence_id",
        "sequence_local_reset_detected",
        "status",
        "strictly_monotonic",
        "world_frames",
    )
    for key in exact_keys:
        _require(actual.get(key) == expected[key], f"reference audit mismatch: {expected['sequence_id']}.{key}")
    for key in (
        "duration_s",
        "first_timestamp_s",
        "last_timestamp_s",
        "max_quaternion_norm_error",
        "maximum_timestamp_gap_s",
        "median_rate_hz",
        "timestamp_scale_to_seconds",
    ):
        _same_number(actual.get(key), expected[key], f"{expected['sequence_id']}.{key}")
    _same_vector(actual.get("first_position_m"), expected["first_position_m"], f"{expected['sequence_id']}.first_position_m")
    _same_vector(actual.get("first_quaternion_xyzw"), expected["first_quaternion_xyzw"], f"{expected['sequence_id']}.first_quaternion_xyzw")


def _topic_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CaversStage1VerificationError(f"invalid rosbag metadata: {path}: {error}") from error
    _require(isinstance(value, dict), f"rosbag metadata root is invalid: {path}")
    info = value.get("rosbag2_bagfile_information")
    _require(isinstance(info, dict), f"rosbag info missing: {path}")
    topics = info.get("topics_with_message_count")
    _require(isinstance(topics, list), f"rosbag topic inventory missing: {path}")
    output: list[dict[str, Any]] = []
    for row in topics:
        if not isinstance(row, dict) or not isinstance(row.get("topic_metadata"), dict):
            continue
        metadata = row["topic_metadata"]
        output.append({"name": metadata.get("name"), "type": metadata.get("type"), "message_count": row.get("message_count")})
    return output


def _verify_reference(runtime_root: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    audit = _mapping_json(runtime_root / "cavers_reference_audit.json")
    rows = audit.get("rows")
    _require(isinstance(rows, list) and len(rows) == len(CANDIDATE_SEQUENCES), "reference row count mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get("sequence_id"), str), "invalid reference row")
        _require(row["sequence_id"] not in by_id, "duplicate reference sequence")
        by_id[row["sequence_id"]] = row
    _require(set(by_id) == set(CANDIDATE_SEQUENCES), "reference candidate inventory mismatch")
    expected_rows: dict[str, dict[str, Any]] = {}
    source_root = data_root / "source_metadata" / "sequences"
    for sequence_id in CANDIDATE_SEQUENCES:
        sequence_root = source_root / sequence_id
        expected = _audit_gt(sequence_root / "GT_ODOM" / "data.csv", sequence_id)
        _require(expected["status"] == "PASS", f"independent GT 6DoF audit failed: {sequence_id}")
        topics = _topic_rows(sequence_root / "rosbag_metadata.yaml")
        topic_by_name = {row["name"]: row for row in topics}
        optitrack = topic_by_name.get("/spaceuma/optitrack/odom")
        _require(
            isinstance(optitrack, dict)
            and optitrack.get("type") == "nav_msgs/msg/Odometry"
            and isinstance(optitrack.get("message_count"), int)
            and not isinstance(optitrack.get("message_count"), bool)
            and optitrack["message_count"] > 0,
            f"direct nonempty OptiTrack topic missing: {sequence_id}",
        )
        _compare_reference_row(by_id[sequence_id], expected)
        _require(by_id[sequence_id].get("rosbag_optitrack_topic_present") is True, f"OptiTrack topic audit mismatch: {sequence_id}")
        _require(
            by_id[sequence_id].get("rosbag_optitrack_message_count") == optitrack["message_count"],
            f"OptiTrack message count mismatch: {sequence_id}",
        )
        expected_rows[sequence_id] = expected
    _require(
        audit.get("candidate_count") == 12
        and audit.get("independent_optitrack_6dof_pass_count") == 12
        and audit.get("quaternion_scalar_last") is True
        and audit.get("status") == "PASS",
        "R02 reference aggregate mismatch",
    )
    csv_rows = _csv(
        runtime_root / "cavers_reference_audit.csv",
        REFERENCE_AUDIT_FIELDS,
    )
    _require(len(csv_rows) == 12 and [row.get("sequence_id") for row in csv_rows] == list(CANDIDATE_SEQUENCES), "reference CSV sequence rows mismatch")
    for row in csv_rows:
        json_row = by_id[row["sequence_id"]]
        expected_csv_row = {field: str(json_row[field]) for field in REFERENCE_AUDIT_FIELDS}
        _require(row == expected_csv_row, f"reference CSV/JSON full-row mismatch: {row['sequence_id']}")
    return expected_rows


def _verify_world_frame(runtime_root: Path, reference: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    value = _mapping_json(runtime_root / "cavers_common_world_frame_audit.json")
    same_named_nonreset = (
        {frame for row in reference.values() for frame in row["world_frames"]} == {"map"}
        and all(row["child_frames"] == ["base_link"] for row in reference.values())
        and not any(row["sequence_local_reset_detected"] for row in reference.values())
    )
    _require(same_named_nonreset, "expected same-named, non-reset GT observation is absent")
    # Independent conclusion for the pinned 0.0.1 / revision 18 release: the
    # selected official metadata publishes no cross-recording calibration ID,
    # session ID, or guarantee that the OptiTrack base stayed fixed.  Therefore
    # the common-world proposition is not established even though every file
    # calls its parent frame `map`.
    independently_proven_shared_calibration = False
    expected_status = "PASS" if same_named_nonreset and independently_proven_shared_calibration else "FAIL"
    _require(value.get("same_named_nonreset_frame_observed") is True, "same-frame observation mismatch")
    _require(value.get("shared_calibration_session_proven") is False, "unpublished shared calibration was asserted")
    _require(value.get("common_world_frame_status") == expected_status == "FAIL", "world-frame conclusion was altered")
    _require(value.get("registration_or_trajectory_alignment_used") is False, "world frame used forbidden alignment")
    _require(value.get("sequence_local_reset_detected") is False and value.get("sequence_local_reset_ids") == [], "reset conclusion mismatch")
    _require(value.get("unique_world_frames") == ["map"], "unique world-frame names mismatch")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 12, "world-frame row count mismatch")
    for row in rows:
        _require(isinstance(row, dict), "invalid world-frame row")
        sequence_id = row.get("sequence_id")
        _require(sequence_id in reference, "unknown world-frame sequence")
        _require(
            row.get("world_frame") == "map"
            and row.get("world_frame_source") == reference[sequence_id]["gt_sha256"]
            and row.get("sequence_local_reset_detected") is False
            and row.get("common_world_frame_status") == "FAIL"
            and isinstance(row.get("shared_calibration_evidence"), str)
            and row["shared_calibration_evidence"].startswith("NOT_PUBLISHED"),
            f"world-frame evidence mismatch: {sequence_id}",
        )
    return value


def _quaternion_matrix(quaternion: Sequence[float]) -> list[list[float]]:
    x, y, z, w = quaternion
    norm_sq = x * x + y * y + z * z + w * w
    _require(norm_sq > 0.0 and math.isfinite(norm_sq), "invalid extrinsic quaternion")
    scale = 2.0 / norm_sq
    return [
        [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
        [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
        [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
    ]


def _parse_extrinsic(path: Path, sequence_id: str) -> dict[str, Any]:
    rows = _csv(path)
    candidates: dict[str, dict[str, Any]] = {}
    for csv_line_number, row in enumerate(rows, 2):
        child = row.get("Child_Frame_ID", "")
        if row.get("Frame_ID") != "base_link" or child != "velodyne":
            continue
        translation = [_float_field(row, name, sequence_id) for name in ("TX", "TY", "TZ")]
        raw_quaternion = [_float_field(row, name, sequence_id) for name in ("QX", "QY", "QZ", "QW")]
        raw_norm = math.sqrt(sum(value * value for value in raw_quaternion))
        norm_error = abs(raw_norm - 1.0)
        _require(raw_norm > 0.0 and norm_error <= 1e-6, f"TF quaternion exceeds export-rounding tolerance: {sequence_id}")
        quaternion = [value / raw_norm for value in raw_quaternion]
        identity = _compact_sha256(
            {
                "parent_frame": row.get("Frame_ID"),
                "child_frame": child,
                "translation_m": translation,
                "quaternion_xyzw": quaternion,
            }
        )
        candidates[identity] = {
            "child_frame": child,
            "composition_rule": "T_map_lidar(t) = T_map_rig(t) @ T_rig_lidar",
            "csv_line_number": csv_line_number,
            "direction": f"T_{row.get('Frame_ID')}_{child}",
            "direction_test_pass": True,
            "parent_frame": row.get("Frame_ID"),
            "quaternion_xyzw": quaternion,
            "quaternion_xyzw_raw": raw_quaternion,
            "quaternion_raw_norm": raw_norm,
            "quaternion_raw_norm_error": norm_error,
            "quaternion_rounding_tolerance": 1e-6,
            "rotation_matrix": _quaternion_matrix(quaternion),
            "timestamp": row.get("Timestamp", ""),
            "translation_m": translation,
            "units": {"rotation": "dimensionless", "translation": "m"},
        }
    _require(len(candidates) == 1, f"rig-to-VLP16 transform missing/ambiguous: {sequence_id}")
    selected = next(iter(candidates.values()))
    _require(selected["parent_frame"] == "base_link" and selected["child_frame"] == "velodyne", f"extrinsic direction mismatch: {sequence_id}")
    selected.update(
        {
            "sequence_id": sequence_id,
            "source_file": str(path),
            "source_sha256": _sha256(path),
            "source_topic": "/tf_static",
        }
    )
    return selected


def _compare_extrinsic(actual: Any, expected: Mapping[str, Any], sequence_id: str) -> None:
    _require(isinstance(actual, dict), f"missing transform object: {sequence_id}")
    for key in (
        "child_frame",
        "composition_rule",
        "csv_line_number",
        "direction",
        "direction_test_pass",
        "parent_frame",
        "sequence_id",
        "source_file",
        "source_sha256",
        "source_topic",
        "timestamp",
        "units",
    ):
        _require(actual.get(key) == expected[key], f"extrinsic provenance mismatch: {sequence_id}.{key}")
    _same_vector(actual.get("translation_m"), expected["translation_m"], f"{sequence_id}.translation_m")
    _same_vector(actual.get("quaternion_xyzw"), expected["quaternion_xyzw"], f"{sequence_id}.quaternion_xyzw")
    _same_vector(actual.get("quaternion_xyzw_raw"), expected["quaternion_xyzw_raw"], f"{sequence_id}.quaternion_xyzw_raw")
    for key in ("quaternion_raw_norm", "quaternion_raw_norm_error", "quaternion_rounding_tolerance"):
        _same_number(actual.get(key), expected[key], f"{sequence_id}.{key}", atol=1e-12)
    matrix = actual.get("rotation_matrix")
    _require(isinstance(matrix, list) and len(matrix) == 3, f"invalid extrinsic matrix: {sequence_id}")
    for index, expected_row in enumerate(expected["rotation_matrix"]):
        _same_vector(matrix[index], expected_row, f"{sequence_id}.rotation_matrix[{index}]")


def _verify_extrinsics(runtime_root: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    audit = _mapping_json(runtime_root / "cavers_extrinsic_audit.json")
    rows = audit.get("rows")
    _require(isinstance(rows, list) and len(rows) == 12, "extrinsic row count mismatch")
    by_id = {row.get("sequence_id"): row for row in rows if isinstance(row, dict)}
    _require(set(by_id) == set(CANDIDATE_SEQUENCES), "extrinsic sequence inventory mismatch")
    expected: dict[str, dict[str, Any]] = {}
    for sequence_id in CANDIDATE_SEQUENCES:
        path = data_root / "source_metadata" / "sequences" / sequence_id / "TF_STATIC" / "data.csv"
        transform = _parse_extrinsic(path, sequence_id)
        expected[sequence_id] = transform
        _require(by_id[sequence_id].get("status") == "PASS", f"extrinsic status mismatch: {sequence_id}")
        _compare_extrinsic(by_id[sequence_id].get("transform"), transform, sequence_id)
    _require(
        audit.get("status") == "PASS"
        and audit.get("identical_cross_sequence_transform_count") == 1
        and audit.get("extrinsic_uncertainty") == "UNKNOWN"
        and audit.get("nominal_cad_transform_is_zero_uncertainty") is False,
        "extrinsic aggregate/uncertainty mismatch",
    )
    chain = _mapping_json(runtime_root / "cavers_transform_chain_manifest.json")
    _require(
        chain.get("composition") == "T_map_lidar(t) = T_map_rig(t) @ T_rig_lidar"
        and chain.get("direct_optitrack_gt") is True
        and chain.get("status") == "PASS",
        "transform chain conclusion mismatch",
    )
    chain_extrinsics = chain.get("extrinsics")
    _require(isinstance(chain_extrinsics, dict) and set(chain_extrinsics) == set(CANDIDATE_SEQUENCES), "transform-chain sequence inventory mismatch")
    for sequence_id in CANDIDATE_SEQUENCES:
        _compare_extrinsic(chain_extrinsics[sequence_id], expected[sequence_id], sequence_id)
    return expected


def _audit_lidar_timestamps(path: Path, sequence_id: str) -> dict[str, Any]:
    rows = _csv(path)
    raw = [_float_field(row, "Timestamp", sequence_id) for row in rows]
    _require(len(raw) >= 2, f"LiDAR timestamp index is incomplete: {sequence_id}")
    scale = 1e-9 if statistics.median(raw) > 1e12 else 1.0
    times = [value * scale for value in raw]
    return {
        "first_timestamp_s": times[0],
        "last_timestamp_s": times[-1],
        "row_count": len(times),
        "strictly_monotonic": all(right > left for left, right in zip(times, times[1:])),
        "timestamp_resolution_s": scale,
    }


def _verify_time(runtime_root: Path, data_root: Path, reference: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    value = _mapping_json(runtime_root / "cavers_time_sync_audit.json")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 12, "time-audit row count mismatch")
    by_id = {row.get("sequence_id"): row for row in rows if isinstance(row, dict)}
    _require(set(by_id) == set(CANDIDATE_SEQUENCES), "time-audit sequence inventory mismatch")
    for sequence_id in CANDIDATE_SEQUENCES:
        sequence_root = data_root / "source_metadata" / "sequences" / sequence_id
        expected = _audit_lidar_timestamps(sequence_root / "VELODYNE_CLOUD" / "data.csv", sequence_id)
        _require(expected["strictly_monotonic"] is True, f"LiDAR timestamps not monotonic: {sequence_id}")
        topics = {row["name"]: row for row in _topic_rows(sequence_root / "rosbag_metadata.yaml")}
        tf_static = topics.get("/tf_static")
        velodyne = topics.get("/spaceuma/velodyne_points")
        _require(
            isinstance(tf_static, dict)
            and tf_static.get("type") == "tf2_msgs/msg/TFMessage"
            and isinstance(tf_static.get("message_count"), int)
            and not isinstance(tf_static.get("message_count"), bool)
            and tf_static["message_count"] > 0,
            f"nonempty /tf_static topic missing: {sequence_id}",
        )
        _require(
            isinstance(velodyne, dict)
            and velodyne.get("type") == "sensor_msgs/msg/PointCloud2"
            and isinstance(velodyne.get("message_count"), int)
            and not isinstance(velodyne.get("message_count"), bool)
            and velodyne["message_count"] > 0,
            f"nonempty Velodyne topic missing: {sequence_id}",
        )
        row = by_id[sequence_id]
        actual_lidar = row.get("lidar_timestamp")
        _require(isinstance(actual_lidar, dict), f"LiDAR time audit missing: {sequence_id}")
        for key in ("row_count", "strictly_monotonic"):
            _require(actual_lidar.get(key) == expected[key], f"LiDAR time mismatch: {sequence_id}.{key}")
        for key in ("first_timestamp_s", "last_timestamp_s", "timestamp_resolution_s"):
            _same_number(actual_lidar.get(key), expected[key], f"{sequence_id}.lidar.{key}")
        gt = reference[sequence_id]
        overlap = max(0.0, min(gt["last_timestamp_s"], expected["last_timestamp_s"]) - max(gt["first_timestamp_s"], expected["first_timestamp_s"]))
        _require(overlap > 0.0, f"GT/LiDAR spans do not overlap: {sequence_id}")
        _same_number(row.get("gt_lidar_span_overlap_s"), overlap, f"{sequence_id}.time_overlap")
        _same_number(row.get("maximum_gt_gap_s"), gt["maximum_timestamp_gap_s"], f"{sequence_id}.maximum_gt_gap")
        _same_number(row.get("measured_metadata_offset_s"), abs(gt["first_timestamp_s"] - expected["first_timestamp_s"]), f"{sequence_id}.metadata_offset")
        _same_number(row.get("timestamp_resolution_s"), expected["timestamp_resolution_s"], f"{sequence_id}.timestamp_resolution")
        _require(
            row.get("clock_domains") == ["ROS2 recording timestamp", "message header timestamp", "sensor timestamp"]
            and row.get("hardware_sync_available") is False
            and row.get("shared_recording_clock") is True
            and row.get("tf_static_topic_present") is True
            and row.get("velodyne_topic_present") is True
            and row.get("documented_processing_delay_s") == "UNKNOWN"
            and row.get("documented_sensor_delay_s") == "UNKNOWN"
            and row.get("time_sync_uncertainty") == "UNKNOWN",
            f"time uncertainty/delay semantics mismatch: {sequence_id}",
        )
    _require(
        value.get("status") == "PASS_WITH_DOCUMENTED_LIMITATION"
        and value.get("hardware_level_synchronization") is False
        and value.get("time_sync_uncertainty_status") == "UNKNOWN"
        and value.get("zero_delay_assumed") is False
        and value.get("candidate_interpolation_method") == "linear translation plus quaternion SLERP, gap-limited",
        "time-sync aggregate conclusion mismatch",
    )
    return value


def _verify_uncertainty(runtime_root: Path) -> dict[str, Any]:
    value = _mapping_json(runtime_root / "cavers_stage1_uncertainty_feasibility.json")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 10, "uncertainty row count mismatch")
    by_component = {row.get("component"): row for row in rows if isinstance(row, dict)}
    _require(set(by_component) == set(UNCERTAINTY_COMPONENTS), "uncertainty component inventory mismatch")
    expected_source_shas = {
        "reference": _sha256(runtime_root / "cavers_reference_audit.json"),
        "extrinsic": _sha256(runtime_root / "cavers_extrinsic_audit.json"),
        "time": _sha256(runtime_root / "cavers_time_sync_audit.json"),
    }
    for component, unit in UNCERTAINTY_COMPONENTS.items():
        row = by_component[component]
        _require(
            row.get("value") == "UNKNOWN"
            and row.get("unit") == unit
            and row.get("uncertainty_type") == "UNKNOWN"
            and row.get("status") == "UNKNOWN"
            and isinstance(row.get("evidence_type"), str)
            and bool(row["evidence_type"].strip())
            and isinstance(row.get("source"), str)
            and bool(row["source"].strip()),
            f"R10 UNKNOWN value/type/provenance altered: {component}",
        )
        if component.startswith("OptiTrack") or component == "interpolation uncertainty":
            expected_sha = expected_source_shas["reference"]
        elif component.startswith("CAD extrinsic"):
            expected_sha = expected_source_shas["extrinsic"]
        elif component in {"GT timestamp uncertainty", "LiDAR timestamp uncertainty", "synchronization/delay uncertainty"}:
            expected_sha = expected_source_shas["time"]
        else:
            expected_sha = "NOT_AVAILABLE"
        _require(row.get("source_sha256") == expected_sha, f"R10 source SHA mismatch: {component}")
    _require(
        value.get("status") == "PARTIAL"
        and value.get("all_unknown_values_preserved_as_unknown") is True
        and value.get("nominal_transform_treated_as_zero_uncertainty") is False,
        "R10 feasibility conclusion mismatch",
    )
    csv_rows = _csv(
        runtime_root / "cavers_stage1_uncertainty_feasibility.csv",
        ("component", "value", "unit", "uncertainty_type", "evidence_type", "source", "source_sha256", "status"),
    )
    _require(len(csv_rows) == 10 and {row["component"] for row in csv_rows} == set(UNCERTAINTY_COMPONENTS), "uncertainty CSV inventory mismatch")
    for row in csv_rows:
        expected = by_component[row["component"]]
        _require(all(row[key] == str(expected[key]) for key in row), f"uncertainty CSV/JSON mismatch: {row['component']}")
    return value


def _verify_overlap(runtime_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlap = _mapping_json(runtime_root / "cavers_gt_only_overlap_matrix.json")
    _require(overlap.get("contract") == OVERLAP_CONTRACT, "frozen GT-only overlap threshold changed")
    _require(overlap.get("gate_status") == "NOT_COMPUTABLE", "overlap must be blocked by unproven common world")
    _require(overlap.get("pair_rows") == [], "overlap rows exist despite a closed world-frame gate")
    _require(overlap.get("point_cloud_or_registration_consulted") is False, "overlap consulted forbidden point cloud/registration")
    csv_rows = _csv(
        runtime_root / "cavers_gt_only_overlap_matrix.csv",
        (
            "map_sequence_id",
            "query_sequence_id",
            "overlap_status",
            "total_covered_duration_s",
            "coverage_fraction",
            "eligible_nonoverlapping_5s_interval_count",
            "nearest_distance_median_m",
            "nearest_distance_q95_m",
            "nearest_distance_max_m",
        ),
    )
    _require(csv_rows == [], "overlap CSV has rows despite NOT_COMPUTABLE gate")
    selection = _mapping_json(runtime_root / "cavers_pair_selection.json")
    _require(
        selection.get("eligible_pair_count") == 0
        and selection.get("PRIMARY_CANDIDATE_MAP_QUERY_PAIR") is None
        and selection.get("RESERVE_PAIR_1") is None
        and selection.get("RESERVE_PAIR_2") is None,
        "chosen/reserve pair was altered despite zero eligible pairs",
    )
    _require(selection.get("ranking_rule") == PAIR_RANKING_RULE, "pair ranking rule changed")
    activation = selection.get("activation_policy")
    _require(isinstance(activation, str) and "never registration outcome" in activation, "reserve activation policy changed")
    return overlap, selection


def _normalized_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _scan_json_for_results(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_field(str(key))
            child_path = f"{path}.{key}"
            if normalized in SUSPICIOUS_RESULT_FIELDS:
                findings.append(child_path)
            elif normalized in ZERO_REGISTRATION_FIELDS and (
                isinstance(child, bool) or not isinstance(child, int) or child != 0
            ):
                findings.append(child_path)
            findings.extend(_scan_json_for_results(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_scan_json_for_results(child, f"{path}[{index}]"))
    return findings


def _verify_no_registration(runtime_root: Path, data_root: Path) -> dict[str, Any]:
    attestation = _mapping_json(runtime_root / "NO_ICP_ATTESTATION.json")
    required_counts = (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "estimated_transform_count",
        "registration_execution_count",
        "real_trial_result_count",
    )
    _require(all(attestation.get(key) == 0 and not isinstance(attestation.get(key), bool) for key in required_counts), "NO_ICP attestation has a nonzero count")
    for optional in ("estimated_transform_file_count", "structured_result_scan_error_count"):
        _require(attestation.get(optional, 0) == 0 and not isinstance(attestation.get(optional, 0), bool), f"NO_ICP {optional} is nonzero")
    _require(attestation.get("pass") is True and attestation.get("status") == "PASS", "NO_ICP attestation did not pass")
    _require(attestation.get("estimated_transform_evidence", []) == [] and attestation.get("estimated_transform_files", []) == [], "NO_ICP evidence list is nonempty")
    findings: list[str] = []
    for root in (runtime_root, data_root):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            lower = path.name.lower()
            if any(token in lower for token in SUSPICIOUS_FILENAME_TOKENS):
                findings.append(str(path))
            if "raw_results" in path.parts:
                findings.append(str(path))
            if path.suffix.lower() == ".json":
                value = _json(path)
                findings.extend(f"{path}#{item}" for item in _scan_json_for_results(value))
            elif path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", newline="") as stream:
                    fields = next(csv.reader(stream), [])
                findings.extend(
                    f"{path}#$header.{field}"
                    for field in fields
                    if _normalized_field(field) in SUSPICIOUS_RESULT_FIELDS
                )
    _require(findings == [], f"point-cloud registration result evidence found: {findings[:5]}")
    return attestation


def _verify_eligibility_and_summary(
    runtime_root: Path,
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligibility = _mapping_json(runtime_root / "cavers_stage1_eligibility.json")
    expected_statuses = {
        "R01": "PENDING_SECOND_DATASET",
        "R02": "PASS",
        "R03": "BLOCKED_STAGE1",
        "R04": "FAIL",
        "R05": "PASS_WITH_DOCUMENTED_LIMITATION",
        "R06": "BLOCKED_STAGE1",
        "R07": "BLOCKED_STAGE1",
        "R08": "BLOCKED_STAGE1",
        "R09": "PASS",
        "R10": "PARTIAL",
        "R14": "BLOCKED_STAGE1",
        "GT_ONLY_OVERLAP": "NOT_COMPUTABLE",
    }
    _require(all(eligibility.get(key) == expected for key, expected in expected_statuses.items()), "R02-R05/R10 or overlap eligibility status mismatch")
    for key in (
        "CAVERS_STAGE1_READY",
        "SINGLE_DATASET_PREREGISTRATION_READY",
        "REAL_DATA_RUN_AUTHORIZED",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
    ):
        _require(eligibility.get(key) is False, f"unauthorized/readiness flag changed: {key}")
    for key in (
        "large_lidar_download_count",
        "planned_future_trials",
        "registration_execution_count",
        "rich_snapshot_count",
        "snapshot_count",
        "weak_snapshot_count",
    ):
        _require(eligibility.get(key) == 0 and not isinstance(eligibility.get(key), bool), f"nonzero Stage-1 count: {key}")
    _require(
        set(eligibility.get("primary_hard_blockers", []))
        == {"COMMON_WORLD_FRAME", "GT_ONLY_OVERLAP", "R10_UNCERTAINTY_FEASIBILITY"},
        "hard-blocker set mismatch",
    )
    summary = _mapping_json(runtime_root / "cavers_stage1_summary.json")
    _require(summary.get("cavers_stage1_ready") is False, "summary readiness was altered")
    _require(summary.get("eligibility") == eligibility, "summary eligibility copy mismatch")
    _require(summary.get("pair_selection") == selection, "summary pair selection copy mismatch")
    _require(isinstance(summary.get("answers"), list) and len(summary["answers"]) == 26, "summary must contain 26 answers")
    conclusion = summary.get("final_conclusion")
    _require(isinstance(conclusion, str) and "未通过" in conclusion and "不下载 VLP-16" in conclusion, "summary fail-closed conclusion mismatch")
    markdown = (runtime_root / "cavers_stage1_summary.md").read_text(encoding="utf-8")
    for token in (
        "CAVERS_STAGE1_READY=false",
        "weak_snapshot_count=0",
        "rich_snapshot_count=0",
        "snapshot_count=0",
        "planned_future_trials=0",
        "registration_execution_count=0",
        "REAL_DATA_RUN_AUTHORIZED=false",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false",
    ):
        _require(token in markdown, f"summary Markdown omits {token}")
    return eligibility, summary


def _verify_sequence_inventory(runtime_root: Path, reference: Mapping[str, Mapping[str, Any]]) -> None:
    value = _mapping_json(runtime_root / "cavers_sequence_inventory.json")
    _require(value.get("candidate_sequences") == list(CANDIDATE_SEQUENCES), "candidate sequence inventory changed")
    _require(value.get("excluded_no_independent_gt") == list(EXCLUDED_NO_GT_SEQUENCES), "no-GT exclusions changed")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 12, "sequence inventory row count mismatch")
    for row in rows:
        _require(isinstance(row, dict), "invalid sequence inventory row")
        sequence_id = row.get("sequence_id")
        _require(sequence_id in reference, "unknown sequence inventory row")
        expected = reference[sequence_id]
        _require(
            row.get("gt_available") is True
            and row.get("velodyne_present") is True
            and row.get("transform_calibration_present") is True
            and row.get("timestamp_index_available") is True
            and row.get("rig_velodyne_transform_available") is True
            and row.get("reference_frame") == "map"
            and row.get("room") == "SALA_DEL_DIABLO"
            and row.get("platform_type") == ("DIABLO_ROBOT" if "diablo" in sequence_id else "HANDHELD_RIG"),
            f"sequence inventory capability mismatch: {sequence_id}",
        )
        _same_number(row.get("gt_duration_s"), expected["duration_s"], f"{sequence_id}.inventory.duration")
        _same_number(row.get("gt_rate_hz"), expected["median_rate_hz"], f"{sequence_id}.inventory.rate")
    csv_rows = _csv(runtime_root / "cavers_sequence_inventory.csv")
    _require(len(csv_rows) == 12 and [row.get("sequence_id") for row in csv_rows] == list(CANDIDATE_SEQUENCES), "sequence inventory CSV mismatch")


def verify_cavers_stage1(
    *,
    repository: Path,
    data_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    """Verify a completed CAVERS Stage-1 runtime and live small evidence.

    A valid verification raises no exception even though the scientific
    readiness result is expected to be ``false``.  Any integrity or semantic
    mutation raises :class:`CaversStage1VerificationError`.
    """

    for label, original in (
        ("repository", Path(repository)),
        ("data root", Path(data_root)),
        ("runtime root", Path(runtime_root)),
    ):
        _require(not original.is_symlink(), f"{label} cannot be a symlink")
    repository = Path(repository).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    runtime_root = Path(runtime_root).resolve(strict=True)
    _require(repository.is_dir() and data_root.is_dir() and runtime_root.is_dir(), "verification roots must be directories")

    actual_runtime_files = _runtime_file_names(runtime_root)
    optional_runtime_files = actual_runtime_files - REQUIRED_RUNTIME_FILES
    _require(
        REQUIRED_RUNTIME_FILES <= actual_runtime_files
        and all(
            name == RESUME_ENVIRONMENT_FILE
            or RESUME_ENVIRONMENT_EXTRA_PATTERN.fullmatch(name) is not None
            for name in optional_runtime_files
        ),
        "required/optional runtime inventory mismatch",
    )
    _verify_sha256sums(runtime_root, actual_runtime_files)
    cleanup = _mapping_json(runtime_root / "cleanup_report.json")
    cavers_cleanup = _mapping_json(runtime_root / "cavers_cleanup_report.json")
    _require(cleanup == cavers_cleanup, "cleanup_report and cavers_cleanup_report JSON differ")
    _require(
        (runtime_root / "cleanup_report.json").read_bytes()
        == (runtime_root / "cavers_cleanup_report.json").read_bytes(),
        "cleanup report canonical bytes differ",
    )

    data_rows, maximum_size = _scan_data_files(data_root)
    _verify_manifest(runtime_root, data_rows, repository)
    download = _verify_download_manifest(runtime_root, data_rows, maximum_size)
    official, archive_rows = _verify_official_source(runtime_root, data_root)
    _verify_small_members(official, archive_rows, data_root, runtime_root)
    reference = _verify_reference(runtime_root, data_root)
    _verify_sequence_inventory(runtime_root, reference)
    world = _verify_world_frame(runtime_root, reference)
    extrinsics = _verify_extrinsics(runtime_root, data_root)
    time_audit = _verify_time(runtime_root, data_root, reference)
    uncertainty = _verify_uncertainty(runtime_root)
    overlap, selection = _verify_overlap(runtime_root)
    attestation = _verify_no_registration(runtime_root, data_root)
    eligibility, summary = _verify_eligibility_and_summary(runtime_root, selection)

    backend_path = repository / "frozen_assets" / "backend_parameter_contract.json"
    _require(_sha256(backend_path) == BACKEND_PARAMETER_SHA256, "live backend parameter contract SHA mismatch")
    _require(eligibility["R02"] == "PASS", "independent R02 recomputation failed")
    _require(world["common_world_frame_status"] == "FAIL", "independent common-world recomputation failed")
    _require(all(value["parent_frame"] == "base_link" and value["child_frame"] == "velodyne" for value in extrinsics.values()), "independent extrinsic direction recomputation failed")
    _require(time_audit["status"] == "PASS_WITH_DOCUMENTED_LIMITATION", "independent time-chain recomputation failed")
    _require(uncertainty["status"] == "PARTIAL", "independent R10 recomputation failed")
    _require(overlap["gate_status"] == "NOT_COMPUTABLE", "overlap fail-closed gate mismatch")

    return {
        "CAVERS_STAGE1_VERIFICATION_PASS": True,
        "artifact_integrity_pass": True,
        "cavers_stage1_ready": False,
        "checked_gt_sequence_count": len(reference),
        "checked_small_file_count": len(data_rows),
        "eligible_pair_count": selection["eligible_pair_count"],
        "maximum_materialized_file_bytes": maximum_size,
        "no_registration_pass": attestation["pass"],
        "overlap_gate_status": overlap["gate_status"],
        "r02_status": eligibility["R02"],
        "r03_status": eligibility["R03"],
        "r04_status": eligibility["R04"],
        "r05_status": eligibility["R05"],
        "r10_status": eligibility["R10"],
        "recorded_materialized_bytes": download["materialized_bytes"],
        "summary": summary,
        "verification_pass": True,
        "world_frame_status": world["common_world_frame_status"],
    }


__all__ = [
    "CANDIDATE_SEQUENCES",
    "CaversStage1VerificationError",
    "MAX_STAGE1_FILE_BYTES",
    "OVERLAP_CONTRACT",
    "RESUME_ENVIRONMENT_EXTRA_PATTERN",
    "RESUME_ENVIRONMENT_FILE",
    "RESUME_POLICY",
    "REQUIRED_RUNTIME_FILES",
    "verify_cavers_stage1",
]
