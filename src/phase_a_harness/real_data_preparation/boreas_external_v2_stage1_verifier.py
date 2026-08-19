"""Independent verifier for the Boreas external-validation v2 Stage-1 freeze.

The verifier intentionally redeclares the scientific and serialization
contracts.  In particular, it does not call the v2 producer, its overlap
implementation, or ``stage1_gt_overlap.compute_stage1_gt_overlap``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .io import canonical_json_bytes, compact_sha256, sha256_file
from .public_data_v1_closure_verifier import verify_public_data_v1_closure


EXPECTED_BRANCH = "prep/public-data-external-validation-v2-boreas-stage1"
STAGE2_PREPARATION_BRANCH = "run/boreas-v2-stage2-data-preparation"
BOREAS_V1_COMMIT = "d304ea7a2a201aff21b48e5039422b63b0eb6e99"
PCL_EXTERNAL_BUNDLE = Path("/tmp/synthetic_confirmatory_v2_pcl_v3_requalification")
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
ASSOCIATION_SOURCE_SHA256 = "458190c26d8640353004663474adb64305d1193d89ff0dc68f71f404de8f175d"
GAP_AWARE_SOURCE_SHA256 = "c757e576f600ddf0ea4a0111137ee43d839611267a06631359823907fcb7ac42"

BOREAS_V1_HASHES = {
    "SHA256SUMS": "69a3a138617a315a3853dd33a519f9f142bd5bd9d08a3ace65b1dca9ba8d1975",
    "boreas_lidar_extrinsic_provenance.json": "fda2be032b760caefdc8db89619a9d47c7deec1a513b72f7714c01f5791a9cf0",
    "boreas_stage1_eligibility.json": "b76102144ad63e9383ed3027b574ce33c7023eda4b328b6b99621bb8d577f8c7",
    "boreas_stage1_manifest.json": "29c582361eb83392e54d34ea4465ae4fb6b1813c89a33a5fa1eb85e1abbab112",
}

PROTOCOL_HASHES = {
    "public_data_external_validation_analysis_contract_v2.json": "462b3e5c0a9c78d161b7d40ff0187ba258efa42d06863aad1e512ebec0439e4a",
    "public_data_external_validation_eligibility_v2.csv": "2c03a24c21f240bee36ee1db9cbe774057af50a23457ccd3d0cfc124f21cee7a",
    "public_data_external_validation_pair_selection_contract_v2.json": "55547571b47e3f8d29e7e6024815b66c79f33b823c993ce83eb94c7525be1231",
    "public_data_external_validation_protocol_v2.json": "9a367081dba37dd9dee8ca6c2d6ed9d83495475521672e2d9223c269947525d4",
    "public_data_external_validation_protocol_v2.md": "2c6348cecf7b1afe67c444ac5ec178e78b1e29a8f505bb4eb3988a1162c936ed",
}

V1_CLOSURE_COPIES = (
    "public_data_v1_closure_summary.md",
    "public_data_v1_closure.json",
    "public_data_v1_candidate_screening.csv",
    "public_data_v1_evidence_manifest.json",
)

PAYLOAD_FILES = frozenset(
    {
        "NO_ICP_ATTESTATION.json",
        "NO_LIDAR_PAYLOAD_ATTESTATION.json",
        "boreas_v2_evidence_reuse_manifest.json",
        "boreas_v2_gt_only_overlap_matrix.csv",
        "boreas_v2_gt_only_overlap_matrix.json",
        "boreas_v2_pair_selection.json",
        "boreas_v2_pair_selection.md",
        "boreas_v2_primary_pair_lidar_metadata_inventory.csv",
        "boreas_v2_requalification_from_frozen_evidence.json",
        "boreas_v2_stage2_disk_budget.json",
        "boreas_v2_stage2_download_allowlist.csv",
        "boreas_v2_stage2_download_plan.json",
        "public_data_external_validation_analysis_contract_v2.json",
        "public_data_external_validation_eligibility_v2.csv",
        "public_data_external_validation_pair_selection_contract_v2.json",
        "public_data_external_validation_protocol_v2.json",
        "public_data_external_validation_protocol_v2.md",
        "public_data_v1_candidate_screening.csv",
        "public_data_v1_closure.json",
        "public_data_v1_closure_summary.md",
        "public_data_v1_evidence_manifest.json",
        "public_data_v2_stage1_eligibility.json",
        "public_data_v2_stage1_summary.json",
        "public_data_v2_stage1_summary.md",
        "test_baseline_status.json",
        "test_baseline_status.md",
    }
)
REQUIRED_FILES = PAYLOAD_FILES | {"frozen_manifest.json", "SHA256SUMS"}

OVERLAP_FIELDS = (
    "map_sequence_id",
    "query_sequence_id",
    "query_duration_s",
    "covered_duration_s",
    "coverage_fraction",
    "contiguous_covered_interval_count",
    "eligible_nonoverlapping_5s_intervals",
    "nearest_distance_median_m",
    "nearest_distance_q95_m",
    "nearest_distance_max_m",
    "map_calibration_sha256",
    "query_calibration_sha256",
    "fixed_ENU_ref_status",
    "pair_eligibility_status",
    "exclusion_reason",
)
METADATA_FIELDS = ("sequence_id", "key", "timestamp_us", "last_modified", "size_bytes")
ALLOWLIST_FIELDS = (
    "selection_role",
    "sequence_id",
    "key",
    "timestamp_us",
    "last_modified",
    "size_bytes",
    "selection_reason",
)
ELIGIBLE_FIELDS = (
    "sequence_id",
    "status",
    "gps_post_process_available",
    "lidar_poses_available",
    "calibration_available",
    "row_count",
    "duration_s",
    "median_rate_hz",
    "maximum_gap_s",
    "first_timestamp_s",
    "last_timestamp_s",
    "reference_frame",
    "pose_sha256",
)
POSE_HEADER = (
    "GPSTime",
    "easting",
    "northing",
    "altitude",
    "vel_east",
    "vel_north",
    "vel_up",
    "roll",
    "pitch",
    "heading",
    "angvel_z",
    "angvel_y",
    "angvel_x",
)

OVERLAP_CONTRACT = {
    "maximum_native_gap_s": 0.2,
    "min_contiguous_covered_duration_s": 5.0,
    "min_coverage_fraction": 0.60,
    "min_eligible_nonoverlapping_5s_intervals": 30,
    "min_total_covered_duration_s": 150.0,
    "radius_m": 5.0,
    "resample_rate_hz": 1.0,
}
PAIR_THRESHOLDS = {
    "GT_OVERLAP_RADIUS_M": 5.0,
    "GT_RESAMPLE_RATE_HZ": 1.0,
    "MAX_NATIVE_REFERENCE_GAP_S": 0.2,
    "MIN_CONTIGUOUS_COVERED_DURATION_S": 5.0,
    "MIN_COVERAGE_FRACTION": 0.6,
    "MIN_ELIGIBLE_NONOVERLAPPING_5S_INTERVALS": 30,
    "MIN_TOTAL_COVERED_DURATION_S": 150.0,
}
RANKING_RULE = [
    {"direction": "descending", "field": "covered_duration_s", "priority": 1},
    {"direction": "descending", "field": "coverage_fraction", "priority": 2},
    {
        "direction": "descending",
        "field": "eligible_nonoverlapping_5s_intervals",
        "priority": 3,
    },
    {"direction": "ascending", "field": "nearest_distance_q95_m", "priority": 4},
    {
        "direction": "lexicographic_ascending",
        "field": "map_sequence_id",
        "priority": 5,
    },
    {
        "direction": "lexicographic_ascending",
        "field": "query_sequence_id",
        "priority": 6,
    },
]
RESERVE_ALLOWED = [
    "download_object_permanently_missing",
    "checksum_mismatch",
    "file_corruption",
    "GT_file_corruption",
    "official_object_withdrawn_for_primary_pair",
]
RESERVE_DISALLOWED = [
    "ICP_error_too_large",
    "weak_rich_result_unfavorable",
    "Open3D_PCL_disagreement",
    "correlation_not_significant",
    "publication_result_unfavorable",
]


class BoreasExternalV2Stage1VerificationError(RuntimeError):
    """The v2 closure, its evidence, or its scientific meaning was altered."""


def _fail(message: str) -> None:
    raise BoreasExternalV2Stage1VerificationError(message)


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: {actual!r} != {expected!r}")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"unsafe relative path: {value}")
    return path


def _load_json(path: Path, *, canonical: bool = True) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoreasExternalV2Stage1VerificationError(f"invalid JSON: {path}") from error
    if canonical and canonical_json_bytes(value) != path.read_bytes():
        _fail(f"JSON is not canonical: {path}")
    return value


def _csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _same(reader.fieldnames, list(fields), f"CSV header {path.name}")
        return list(reader)


def _string_row(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, str]:
    return {field: "" if row.get(field) is None else str(row.get(field)) for field in fields}


def _payload_digest(value: Mapping[str, Any], field: str) -> str:
    return compact_sha256({key: item for key, item in value.items() if key != field})


def _parse_sums(root: Path, *, exact_files: set[str] | frozenset[str]) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        _fail(f"SHA256SUMS is absent or unsafe: {root}")
    rows: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _fail(f"malformed SHA256SUMS row in {root}: {line!r}")
        digest, name = match.groups()
        relative = _safe_relative(name)
        if name in rows or name == "SHA256SUMS":
            _fail(f"duplicate or recursive SHA256SUMS path: {name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            _fail(f"SHA256 closure mismatch: {path}")
        rows[name] = digest
    _same(set(rows), set(exact_files) - {"SHA256SUMS"}, f"SHA256SUMS inventory {root}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _same(actual, set(exact_files), f"exact file closure {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        _fail(f"symlink in immutable closure: {root}")
    return rows


def _verify_runtime_closure(root: Path) -> Mapping[str, Any]:
    if root.is_symlink() or not root.is_dir():
        _fail("v2 runtime/frozen root is absent or a symlink")
    _parse_sums(root, exact_files=REQUIRED_FILES)
    manifest = _load_json(root / "frozen_manifest.json")
    unsigned = dict(manifest)
    recorded_root = unsigned.pop("manifest_root_sha256", None)
    _same(recorded_root, compact_sha256(unsigned), "v2 manifest root SHA")
    _same(
        manifest.get("schema_version"),
        "public_data_external_validation_v2_boreas_stage1_manifest_v1",
        "v2 manifest schema",
    )
    _same(manifest.get("boreas_v1_commit"), BOREAS_V1_COMMIT, "v1 commit binding")
    _same(manifest.get("boreas_v1_evidence_count"), 84, "v1 evidence count binding")
    _same(
        manifest.get("backend_parameter_contract_sha256"),
        BACKEND_PARAMETER_SHA256,
        "backend parameter SHA binding",
    )
    payload = manifest.get("payload")
    if not isinstance(payload, list) or len(payload) != len(PAYLOAD_FILES):
        _fail("manifest payload row count mismatch")
    _same([row.get("path") for row in payload], sorted(PAYLOAD_FILES), "manifest payload order")
    for row in payload:
        name = row.get("path")
        path = root / str(name)
        _same(row.get("sha256"), sha256_file(path), f"manifest payload SHA {name}")
        _same(row.get("size_bytes"), path.stat().st_size, f"manifest payload size {name}")
    _same(
        manifest.get("eligibility_sha256"),
        sha256_file(root / "public_data_v2_stage1_eligibility.json"),
        "manifest eligibility SHA",
    )
    producer = manifest.get("producer_commit")
    if not isinstance(producer, str) or re.fullmatch(r"[0-9a-f]{40}", producer) is None:
        _fail("manifest producer commit is invalid")
    return manifest


def _verify_protocols(repository: Path, root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocols = repository / "protocols"
    for name, expected_sha in PROTOCOL_HASHES.items():
        source = protocols / name
        copied = root / name
        _same(sha256_file(source), expected_sha, f"normative protocol SHA {name}")
        _same(copied.read_bytes(), source.read_bytes(), f"copied protocol bytes {name}")
    _same(manifest.get("protocol_sha256"), PROTOCOL_HASHES, "manifest protocol SHA inventory")
    _same(
        sha256_file(repository / "frozen_assets/backend_parameter_contract.json"),
        BACKEND_PARAMETER_SHA256,
        "backend parameter contract",
    )
    _same(
        sha256_file(repository / "src/phase_a_harness/common_association_analysis.py"),
        ASSOCIATION_SOURCE_SHA256,
        "association implementation",
    )
    _same(
        sha256_file(repository / "src/phase_a_harness/real_data_preparation/stage1_gt_overlap.py"),
        GAP_AWARE_SOURCE_SHA256,
        "gap-aware implementation",
    )

    protocol = _load_json(protocols / "public_data_external_validation_protocol_v2.json")
    _same(
        protocol.get("protocol_payload_sha256"),
        _payload_digest(protocol, "protocol_payload_sha256"),
        "protocol payload digest",
    )
    for key, expected in (
        ("role", "SUPPLEMENTARY_EXTERNAL_GENERALIZATION"),
        ("primary_metrological_evidence", False),
        ("public_dataset_count_required", 1),
        ("selected_dataset", "Boreas"),
        ("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED", False),
        ("REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED", False),
        ("PUBLIC_DATA_V2_RUN_AUTHORIZED", False),
        ("REAL_REGISTRATION_AUTHORIZED", False),
        ("registration_execution_count", 0),
        ("downloaded_lidar_payload_count", 0),
    ):
        _same(protocol.get(key), expected, f"protocol {key}")
    requirements = protocol.get("eligibility_requirements")
    _same(set(requirements or {}), {f"E{index:02d}" for index in range(1, 15)}, "E01-E14 set")
    e02 = requirements["E02"]
    requirement_text = str(e02.get("requirement", ""))
    for token in ("query-to-target", "ICP", "scan matching", "LiDAR odometry", "SLAM"):
        if token not in requirement_text:
            _fail(f"E02 trajectory-independence token missing: {token}")
    static = e02.get("historical_lidar_assisted_static_extrinsic", {})
    _same(static.get("allowed"), True, "E02 historical LiDAR-assisted allowance")
    _same(static.get("all_conditions_required"), True, "E02 all-conditions rule")
    conditions = static.get("conditions")
    _same(
        [row.get("condition_id") for row in conditions or []],
        [f"E02-X{index:02d}" for index in range(1, 11)],
        "E02 ten limitations",
    )
    limitation_text = " ".join(str(row.get("requirement", "")) for row in conditions)
    for token in ("not re-estimated", "same frozen byte-identical matrix", "LiDAR-assisted", "UNKNOWN", "external trend/generalization"):
        if token not in limitation_text:
            _fail(f"E02 limitation missing: {token}")
    _same(
        e02.get("boreas_adjudication_when_all_conditions_evidenced"),
        "PASS_WITH_DOCUMENTED_LIMITATION",
        "E02 permitted adjudication",
    )

    analysis = _load_json(protocols / "public_data_external_validation_analysis_contract_v2.json")
    _same(
        analysis.get("contract_payload_sha256"),
        _payload_digest(analysis, "contract_payload_sha256"),
        "analysis contract digest",
    )
    _same(set(analysis.get("hypotheses", {})), {f"V2-H{i}" for i in range(1, 7)}, "V2 hypotheses")
    _same(
        analysis["hypotheses"]["V2-H4"]["association_definition"].get("file_sha256"),
        ASSOCIATION_SOURCE_SHA256,
        "association contract SHA",
    )
    _same(
        analysis["hypotheses"]["V2-H5"].get("unknown_imputed_as_zero"),
        False,
        "analysis UNKNOWN guard",
    )
    _same(
        analysis["hypotheses"]["V2-H6"].get("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED"),
        False,
        "analysis mainline boundary",
    )

    pair = _load_json(protocols / "public_data_external_validation_pair_selection_contract_v2.json")
    _same(pair.get("contract_payload_sha256"), _payload_digest(pair, "contract_payload_sha256"), "pair contract digest")
    _same(pair.get("eligibility_thresholds"), PAIR_THRESHOLDS, "pair thresholds")
    _same(pair.get("pair_record_fields"), list(OVERLAP_FIELDS), "pair fields")
    _same(pair["pair_selection"].get("ranking"), RANKING_RULE, "pair ranking")
    _same(pair["directed_pair_contract"].get("same_sequence_pair_allowed"), False, "same-sequence prohibition")
    _same(pair["directed_pair_contract"].get("expected_pair_count_for_29_sequences"), 812, "directed-pair count")
    reserve = pair.get("reserve_pair_activation", {})
    _same(reserve.get("allowed_only_for_infrastructure_failure"), True, "reserve activation scope")
    _same(reserve.get("allowed_reasons"), RESERVE_ALLOWED, "reserve allowed reasons")
    _same(reserve.get("disallowed_reasons"), RESERVE_DISALLOWED, "reserve disallowed reasons")
    _same(
        pair["gap_aware_overlap_implementation"].get("file_sha256"),
        GAP_AWARE_SOURCE_SHA256,
        "pair gap-aware SHA",
    )

    eligibility_rows = _csv_rows(
        protocols / "public_data_external_validation_eligibility_v2.csv",
        ("requirement_id", "requirement", "mandatory", "required_evidence", "allowed_adjudications", "status"),
    )
    _same([row["requirement_id"] for row in eligibility_rows], [f"E{i:02d}" for i in range(1, 15)], "eligibility CSV IDs")
    if any(row["mandatory"] != "True" or row["status"] != "NOT_EVALUATED" for row in eligibility_rows):
        _fail("protocol eligibility CSV semantics changed")
    return {"protocol": protocol, "pair_contract": pair}


def _verify_boreas_v1_bundle(root: Path) -> None:
    for name, expected in BOREAS_V1_HASHES.items():
        _same(sha256_file(root / name), expected, f"historical Boreas v1 SHA {name}")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _parse_sums(root, exact_files=actual_files)
    eligibility = _load_json(root / "boreas_stage1_eligibility.json")
    for key, expected in (("R02", "FAIL"), ("R10", "PARTIAL"), ("BOREAS_STAGE1_READY", False), ("registration_execution_count", 0)):
        _same(eligibility.get(key), expected, f"historical Boreas v1 {key}")
    extrinsic = _load_json(root / "boreas_lidar_extrinsic_provenance.json")
    for key, expected in (
        ("uses_lidar_pointclouds", True),
        ("all_sequences_byte_identical", True),
        ("query_sequence_used_for_calibration", "UNKNOWN"),
        ("extrinsic_uncertainty", "UNKNOWN"),
        ("frozen_R02_independence_violation", True),
    ):
        _same(extrinsic.get(key), expected, f"historical extrinsic {key}")


def _receipt_path(path: Path) -> Path:
    return path.with_name(path.name + ".receipt.json")


def _authenticate_evidence(
    *, data_root: Path, boreas_v1_root: Path, runtime_root: Path
) -> tuple[dict[str, Path], list[dict[str, Any]], Mapping[str, Any]]:
    manifest = _load_json(boreas_v1_root / "boreas_stage1_manifest.json")
    download = _load_json(boreas_v1_root / "download_manifest.json")
    evidence = manifest.get("data_evidence")
    if not isinstance(evidence, list) or len(evidence) != 84:
        _fail("Boreas v1 data evidence count is not 84")
    _same(evidence, download.get("materialized_files"), "v1 manifest/download evidence")
    paths: dict[str, Path] = {}
    expected_reuse: list[dict[str, Any]] = []
    receipt_count = 0
    for row in evidence:
        relative = row.get("relative_path")
        local = row.get("local_path")
        if not isinstance(relative, str) or not isinstance(local, str):
            _fail("v1 evidence path is invalid")
        _safe_relative(relative)
        path = data_root / relative
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise BoreasExternalV2Stage1VerificationError(f"v1 evidence missing: {relative}") from error
        _same(resolved, Path(local).resolve(strict=True), f"v1 evidence path binding {relative}")
        if path.is_symlink() or data_root not in resolved.parents:
            _fail(f"unsafe v1 evidence path: {relative}")
        _same(path.stat().st_size, row.get("size_bytes"), f"v1 evidence size {relative}")
        _same(sha256_file(path), row.get("sha256"), f"v1 evidence SHA {relative}")
        receipt_path: Path | None = None
        receipt_sha: str | None = None
        if row.get("status") == "DOWNLOADED_ALLOWLISTED_STAGE1_OBJECT":
            receipt_path = _receipt_path(path)
            receipt = _load_json(receipt_path)
            for key, expected in (
                ("local_path", str(path)),
                ("key", row.get("s3_key")),
                ("sha256", row.get("sha256")),
                ("size_bytes", row.get("size_bytes")),
                ("etag", row.get("etag")),
                ("last_modified", row.get("last_modified")),
                ("version_id", row.get("version_id")),
            ):
                _same(receipt.get(key), expected, f"receipt {relative}:{key}")
            receipt_sha = sha256_file(receipt_path)
            receipt_count += 1
        if relative.endswith(".bin") or "/lidar/" in relative:
            _fail("historical Stage-1 evidence contains a LiDAR payload")
        paths[relative] = path
        expected_reuse.append(
            {
                "local_path": str(path),
                "relative_path": relative,
                "receipt_path": None if receipt_path is None else str(receipt_path),
                "receipt_sha256": receipt_sha,
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "source_status": row.get("status"),
            }
        )
    _same(receipt_count, 76, "download receipt count")

    sums_rows: list[dict[str, Any]] = []
    for line in (boreas_v1_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail("malformed v1 Boreas SHA256SUMS row")
        digest, name = match.groups()
        sums_rows.append({"path": name, "sha256": digest, "size_bytes": (boreas_v1_root / name).stat().st_size})
    expected_manifest = {
        "data_evidence_file_count": 84,
        "downloaded_stage1_small_object_count": 76,
        "files": expected_reuse,
        "files_redownloaded": 0,
        "registration_execution_count": 0,
        "v1_frozen_closure": {
            "entry_count": len(sums_rows),
            "rows": sums_rows,
            "sha256sums_sha256": BOREAS_V1_HASHES["SHA256SUMS"],
        },
        "v1_manifest_data_evidence_sha256": compact_sha256(evidence),
        "v1_manifest_sha256": BOREAS_V1_HASHES["boreas_stage1_manifest.json"],
        "v1_sha256sums_sha256": BOREAS_V1_HASHES["SHA256SUMS"],
    }
    _same(
        _load_json(runtime_root / "boreas_v2_evidence_reuse_manifest.json"),
        expected_manifest,
        "v2 evidence reuse manifest",
    )
    return paths, evidence, manifest


def _read_pose(path: Path, sequence_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        if header[: len(POSE_HEADER)] != POSE_HEADER:
            _fail(f"unexpected Boreas pose header: {sequence_id}")
        try:
            values = np.asarray(
                [[float(cell) for cell in row] for row in reader if row], dtype=np.float64
            )
        except ValueError as error:
            raise BoreasExternalV2Stage1VerificationError(
                f"nonnumeric Boreas pose row: {sequence_id}"
            ) from error
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 13
        or not np.isfinite(values).all()
    ):
        _fail(f"invalid Boreas pose matrix: {sequence_id}")
    scale = 1e-6 if float(np.median(values[:, 0])) > 1e12 else 1.0
    times = values[:, 0] * scale
    gaps = np.diff(times)
    if not np.all(gaps > 0.0):
        _fail(f"duplicate or nonmonotonic pose timestamp: {sequence_id}")
    report = {
        "duration_s": float(times[-1] - times[0]),
        "first_timestamp_s": float(times[0]),
        "last_timestamp_s": float(times[-1]),
        "maximum_gap_s": float(np.max(gaps)),
        "median_rate_hz": float(1.0 / np.median(gaps)),
        "pose_sha256": sha256_file(path),
        "reference_frame": "ENU_ref",
        "row_count": int(values.shape[0]),
        "sequence_id": sequence_id,
    }
    return report, np.column_stack((times, values[:, 1:4])), values


def _verify_requalification(
    *, boreas_v1_root: Path, runtime_root: Path
) -> Mapping[str, Any]:
    reference = _load_json(boreas_v1_root / "boreas_reference_trajectory_provenance.json")
    extrinsic = _load_json(boreas_v1_root / "boreas_lidar_extrinsic_provenance.json")
    world = _load_json(boreas_v1_root / "boreas_common_world_frame_audit.json")
    time = _load_json(boreas_v1_root / "boreas_time_sync_audit.json")
    trajectory = {
        "uses_gnss": reference.get("uses_gnss"),
        "uses_imu": reference.get("uses_imu"),
        "uses_wheel_encoder": reference.get("uses_wheel_encoder"),
        "uses_rtx_or_pospac": reference.get("uses_rtx"),
        "uses_lidar": reference.get("uses_lidar"),
        "uses_icp": reference.get("uses_icp"),
        "uses_scan_matching": reference.get("uses_scan_matching"),
    }
    expected_trajectory = {
        "uses_gnss": True,
        "uses_imu": True,
        "uses_wheel_encoder": True,
        "uses_rtx_or_pospac": True,
        "uses_lidar": False,
        "uses_icp": False,
        "uses_scan_matching": False,
    }
    _same(trajectory, expected_trajectory, "independent trajectory backbone")
    static = {
        "fixed": extrinsic.get("common_static_calibration"),
        "identical_across_all_44_sequences": extrinsic.get("all_sequences_byte_identical"),
        "publicly_released": True,
        "calibration_used_lidar_point_clouds": extrinsic.get("uses_lidar_pointclouds"),
        "calibration_date_or_session": extrinsic.get("calibration_date_or_session"),
        "query_participation": extrinsic.get("query_sequence_used_for_calibration"),
        "uncertainty": extrinsic.get("extrinsic_uncertainty"),
        "reestimated_in_v2": False,
        "current_query_target_registration_output_used": False,
    }
    expected_static = {
        "fixed": True,
        "identical_across_all_44_sequences": True,
        "publicly_released": True,
        "calibration_used_lidar_point_clouds": True,
        "calibration_date_or_session": "UNKNOWN",
        "query_participation": "UNKNOWN",
        "uncertainty": "UNKNOWN",
        "reestimated_in_v2": False,
        "current_query_target_registration_output_used": False,
    }
    _same(static, expected_static, "LiDAR-assisted static extrinsic disclosure")
    _same(world.get("status"), "PASS", "fixed ENU_ref status")
    _same(world.get("COMMON_WORLD_FRAME"), "PASS", "fixed ENU_ref declaration")
    _same(time.get("status"), "PASS_WITH_DOCUMENTED_LIMITATION", "time-chain status")
    _same(time.get("time_uncertainty_status"), "UNKNOWN", "time uncertainty")
    _same(time.get("zero_time_uncertainty_assumed"), False, "time uncertainty zero guard")

    uncertainty = _load_json(boreas_v1_root / "boreas_stage1_uncertainty_feasibility.json")
    _same(uncertainty.get("unknown_not_zero"), True, "v1 uncertainty UNKNOWN guard")
    components = {row.get("component"): row for row in uncertainty.get("rows", [])}
    expected_unknown = {
        "orientation uncertainty",
        "Applanix time uncertainty",
        "Velodyne time synchronization uncertainty",
        "T_applanix_lidar translation uncertainty",
        "T_applanix_lidar rotation uncertainty",
        "GT interpolation uncertainty",
        "future deskew uncertainty",
        "future target-map accumulation uncertainty",
    }
    if not expected_unknown.issubset(components):
        _fail("frozen uncertainty inventory lost an UNKNOWN component")
    for name in expected_unknown:
        row = components[name]
        for field in ("value", "uncertainty_type", "status"):
            _same(row.get(field), "UNKNOWN", f"uncertainty {name}:{field}")

    expected = {
        "E02": "PASS_WITH_DOCUMENTED_LIMITATION",
        "boreas_v1_verifier_report": {
            "BOREAS_STAGE1_VERIFICATION_PASS": True,
            "data_evidence_file_count": 84,
            "original_sequence_count": 44,
            "public_gt_sequence_count": 31,
            "registration_execution_count": 0,
        },
        "boreas_v1_verifier_command": (
            "python scripts/verify_boreas_stage1_v1.py --data-root "
            "/home/lj/ZPRM/zero_perturbation_data/boreas_stage1_v1 --runtime-root "
            "/home/lj/ZPRM/frozen_assets/real_data_boreas_stage1_v1"
        ),
        "boreas_v1_verifier_pass_confirmed_before_v2_build": True,
        "common_world_frame_status": "PASS",
        "files_redownloaded": 0,
        "registration_execution_count": 0,
        "static_extrinsic": expected_static,
        "time_chain_status": "PASS_WITH_DOCUMENTED_LIMITATION",
        "trajectory_backbone": expected_trajectory,
        "v1_conclusion": {"BOREAS_STAGE1_READY": False, "R02": "FAIL", "R10": "PARTIAL"},
        "v1_conclusion_preserved": True,
        "v2_role": "SUPPLEMENTARY_EXTERNAL_GENERALIZATION",
        "v2_semantics_versioned": True,
    }
    actual = _load_json(runtime_root / "boreas_v2_requalification_from_frozen_evidence.json")
    _same(actual, expected, "v2 frozen-evidence requalification")
    return actual


def _load_eligible_trajectories(
    *, boreas_v1_root: Path, evidence_paths: Mapping[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    inventory = _load_json(boreas_v1_root / "boreas_remote_sequence_inventory.json")
    original = inventory.get("boreas_original_sequences")
    if not isinstance(original, list) or len(original) != 44:
        _fail("Boreas original sequence inventory is not 44")
    public_sequences = [
        str(row["sequence_id"])
        for row in original
        if row.get("ground_truth_candidate") is True
        and row.get("test_or_training_status") == "TRAIN_PUBLIC_GT"
    ]
    _same(len(public_sequences), 31, "public-GT sequence count")

    all_reports: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, np.ndarray] = {}
    values_by_id: dict[str, np.ndarray] = {}
    for sequence in public_sequences:
        relative = f"stage1_payload/{sequence}/applanix/lidar_poses.csv"
        path = evidence_paths.get(relative)
        if path is None:
            _fail(f"public-GT pose evidence absent: {sequence}")
        report, trajectory, values = _read_pose(path, sequence)
        all_reports[sequence] = report
        trajectories[sequence] = trajectory
        values_by_id[sequence] = values
        source = next(row for row in original if row["sequence_id"] == sequence)
        _same(values.shape[0], source.get("lidar_object_count"), f"pose/object count {sequence}")
        _same(int(round(values[0, 0])), source.get("first_lidar_timestamp"), f"first lidar timestamp {sequence}")
        _same(int(round(values[-1, 0])), source.get("last_lidar_timestamp"), f"last lidar timestamp {sequence}")

    eligible_ids = [
        sequence
        for sequence in public_sequences
        if all_reports[sequence]["maximum_gap_s"] <= OVERLAP_CONTRACT["maximum_native_gap_s"]
    ]
    excluded_ids = [sequence for sequence in public_sequences if sequence not in eligible_ids]
    _same(len(eligible_ids), 29, "eligible reference sequence count")
    _same(len(excluded_ids), 2, "native-gap exclusion count")

    frozen_rows = _csv_rows(boreas_v1_root / "eligible_reference_sequences.csv", ELIGIBLE_FIELDS)
    _same([row["sequence_id"] for row in frozen_rows], eligible_ids, "eligible sequence inventory")
    expected_rows: list[dict[str, str]] = []
    calibration_sha: dict[str, str] = {}
    eligible_reports: list[dict[str, Any]] = []
    for sequence in eligible_ids:
        report = all_reports[sequence]
        expected = {
            "sequence_id": sequence,
            "status": "ELIGIBLE_PUBLIC_GT",
            "gps_post_process_available": True,
            "lidar_poses_available": True,
            "calibration_available": True,
            "row_count": report["row_count"],
            "duration_s": report["duration_s"],
            "median_rate_hz": report["median_rate_hz"],
            "maximum_gap_s": report["maximum_gap_s"],
            "first_timestamp_s": report["first_timestamp_s"],
            "last_timestamp_s": report["last_timestamp_s"],
            "reference_frame": "ENU_ref",
            "pose_sha256": report["pose_sha256"],
        }
        expected_rows.append(_string_row(expected, ELIGIBLE_FIELDS))
        calibration_relative = f"stage1_payload/{sequence}/calib/T_applanix_lidar.txt"
        calibration = evidence_paths.get(calibration_relative)
        if calibration is None:
            _fail(f"eligible calibration absent: {sequence}")
        matrix = np.loadtxt(calibration, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            _fail(f"invalid calibration matrix: {sequence}")
        calibration_sha[sequence] = sha256_file(calibration)
        eligible_reports.append(report)
    _same(frozen_rows, expected_rows, "eligible sequence rows")
    _same(len(set(calibration_sha.values())), 1, "byte-identical eligible extrinsics")

    all_calibration_sha = {
        sha256_file(path)
        for relative, path in evidence_paths.items()
        if relative.endswith("/calib/T_applanix_lidar.txt")
    }
    _same(len(all_calibration_sha), 1, "byte-identical extrinsics across all 44 sequences")
    return (
        sorted(eligible_reports, key=lambda row: row["sequence_id"]),
        {sequence: trajectories[sequence] for sequence in eligible_ids},
        {sequence: values_by_id[sequence] for sequence in eligible_ids},
        calibration_sha,
    )


def _resample_independent(trajectory: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] < 2:
        _fail("invalid overlap trajectory shape")
    if not np.isfinite(value).all() or not np.all(np.diff(value[:, 0]) > 0.0):
        _fail("invalid overlap trajectory values")
    breaks = np.flatnonzero(np.diff(value[:, 0]) > OVERLAP_CONTRACT["maximum_native_gap_s"]) + 1
    step = 1.0 / OVERLAP_CONTRACT["resample_rate_hz"]
    output: list[np.ndarray] = []
    segment_id = 0
    for segment in np.split(value, breaks):
        if segment.shape[0] < 2 or segment[-1, 0] - segment[0, 0] + 1e-12 < step:
            continue
        times = np.arange(segment[0, 0], segment[-1, 0] + step * 1e-9, step)
        positions = np.column_stack(
            [np.interp(times, segment[:, 0], segment[:, column]) for column in (1, 2, 3)]
        )
        output.append(
            np.column_stack((times, positions, np.full(times.size, segment_id, dtype=np.float64)))
        )
        segment_id += 1
    if not output:
        _fail("no trajectory segment can be independently resampled")
    return np.vstack(output)


def _overlap_independent(map_rows: np.ndarray, query_rows: np.ndarray) -> dict[str, Any]:
    distances, _ = cKDTree(map_rows[:, 1:4]).query(query_rows[:, 1:4], k=1)
    covered = distances <= OVERLAP_CONTRACT["radius_m"]
    step = 1.0 / OVERLAP_CONTRACT["resample_rate_hz"]
    intervals: list[dict[str, Any]] = []
    eligible_intervals = 0
    for segment_id in np.unique(query_rows[:, 4]).astype(int):
        indices = np.flatnonzero(query_rows[:, 4] == segment_id)
        flags = covered[indices]
        start: int | None = None
        for local_index, flag in enumerate(np.append(flags, False)):
            if bool(flag) and start is None:
                start = local_index
            elif not bool(flag) and start is not None:
                stop = local_index - 1
                sample_count = stop - start + 1
                duration = sample_count * step
                intervals.append(
                    {
                        "duration_s": float(duration),
                        "end_time": float(query_rows[indices[stop], 0] + step),
                        "sample_count": int(sample_count),
                        "segment_id": int(segment_id),
                        "start_time": float(query_rows[indices[start], 0]),
                    }
                )
                if duration >= OVERLAP_CONTRACT["min_contiguous_covered_duration_s"]:
                    eligible_intervals += int(duration // 5.0)
                start = None
    covered_count = int(np.sum(covered))
    query_count = int(covered.size)
    covered_duration = covered_count * step
    coverage = covered_count / query_count
    passed = (
        covered_duration >= OVERLAP_CONTRACT["min_total_covered_duration_s"]
        and coverage >= OVERLAP_CONTRACT["min_coverage_fraction"]
        and eligible_intervals >= OVERLAP_CONTRACT["min_eligible_nonoverlapping_5s_intervals"]
    )
    return {
        "contract": dict(OVERLAP_CONTRACT),
        "contiguous_covered_intervals": intervals,
        "coverage_fraction": float(coverage),
        "covered_query_count": covered_count,
        "eligible_nonoverlapping_5s_interval_count": int(eligible_intervals),
        "map_resampled_count": int(map_rows.shape[0]),
        "nearest_distance_max_m": float(np.max(distances)),
        "nearest_distance_median_m": float(np.median(distances)),
        "nearest_distance_q95_m": float(np.quantile(distances, 0.95)),
        "overlap_status": "PASS" if passed else "FAIL",
        "query_count": query_count,
        "query_resampled_segment_count": int(np.unique(query_rows[:, 4]).size),
        "total_covered_duration_s": float(covered_duration),
    }


def _compute_all_pairs_independent(
    *, reports: Sequence[Mapping[str, Any]], trajectories: Mapping[str, np.ndarray], calibration_sha: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, np.ndarray]]:
    report_by_id = {str(row["sequence_id"]): row for row in reports}
    sequence_ids = sorted(trajectories)
    resampled = {sequence: _resample_independent(trajectories[sequence]) for sequence in sequence_ids}
    rows: list[dict[str, Any]] = []
    raw_results: dict[tuple[str, str], dict[str, Any]] = {}
    for map_sequence in sequence_ids:
        for query_sequence in sequence_ids:
            if map_sequence == query_sequence:
                continue
            raw = _overlap_independent(resampled[map_sequence], resampled[query_sequence])
            raw_results[(map_sequence, query_sequence)] = raw
            status = raw["overlap_status"]
            rows.append(
                {
                    "map_sequence_id": map_sequence,
                    "query_sequence_id": query_sequence,
                    "query_duration_s": float(report_by_id[query_sequence]["duration_s"]),
                    "covered_duration_s": float(raw["total_covered_duration_s"]),
                    "coverage_fraction": float(raw["coverage_fraction"]),
                    "contiguous_covered_interval_count": len(raw["contiguous_covered_intervals"]),
                    "eligible_nonoverlapping_5s_intervals": int(raw["eligible_nonoverlapping_5s_interval_count"]),
                    "nearest_distance_median_m": float(raw["nearest_distance_median_m"]),
                    "nearest_distance_q95_m": float(raw["nearest_distance_q95_m"]),
                    "nearest_distance_max_m": float(raw["nearest_distance_max_m"]),
                    "map_calibration_sha256": calibration_sha[map_sequence],
                    "query_calibration_sha256": calibration_sha[query_sequence],
                    "fixed_ENU_ref_status": "PASS",
                    "pair_eligibility_status": status,
                    "exclusion_reason": "" if status == "PASS" else "FROZEN_OVERLAP_THRESHOLD_FAILURE",
                }
            )
    _same(len(rows), 812, "independently recomputed directed-pair count")
    ranked = sorted(
        (dict(row) for row in rows if row["pair_eligibility_status"] == "PASS"),
        key=lambda row: (
            -float(row["covered_duration_s"]),
            -float(row["coverage_fraction"]),
            -int(row["eligible_nonoverlapping_5s_intervals"]),
            float(row["nearest_distance_q95_m"]),
            str(row["map_sequence_id"]),
            str(row["query_sequence_id"]),
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return rows, ranked, raw_results, resampled


def _verify_overlap(
    *, runtime_root: Path, reports: Sequence[Mapping[str, Any]], trajectories: Mapping[str, np.ndarray], calibration_sha: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, np.ndarray]]:
    rows, ranked, raw_results, resampled = _compute_all_pairs_independent(
        reports=reports, trajectories=trajectories, calibration_sha=calibration_sha
    )
    frozen = _load_json(runtime_root / "boreas_v2_gt_only_overlap_matrix.json")
    expected = {
        "contract": dict(OVERLAP_CONTRACT),
        "eligible_pair_count": len(ranked),
        "eligible_reference_sequence_count": 29,
        "expected_directed_pair_count": 812,
        "gate_status": "PASS",
        "pair_count": 812,
        "pair_rows": rows,
        "point_cloud_or_registration_consulted": False,
        "status": "PASS",
    }
    _same(frozen, expected, "independently recomputed GT-only overlap matrix")
    csv_rows = _csv_rows(runtime_root / "boreas_v2_gt_only_overlap_matrix.csv", OVERLAP_FIELDS)
    _same(csv_rows, [_string_row(row, OVERLAP_FIELDS) for row in rows], "overlap CSV/JSON projection")
    if not ranked:
        _fail("no pair passes the frozen overlap thresholds")
    return ranked, raw_results, resampled


def _complete_windows(intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(intervals):
        count = int(float(interval["duration_s"]) // 5.0)
        start = float(interval["start_time"])
        for local_index in range(count):
            window_start = start + 5.0 * local_index
            windows.append(
                {
                    "duration_s": 5.0,
                    "end_time_s": window_start + 5.0,
                    "interval_index": interval_index,
                    "window_index": len(windows),
                    "start_time_s": window_start,
                }
            )
    return windows


def _select_timestamps_independent(
    *, map_trajectory: np.ndarray, query_trajectory: np.ndarray,
    map_values: np.ndarray, query_values: np.ndarray,
    query_resampled: np.ndarray, primary_raw: Mapping[str, Any],
) -> dict[str, Any]:
    windows = _complete_windows(primary_raw["contiguous_covered_intervals"])
    _same(
        len(windows),
        primary_raw["eligible_nonoverlapping_5s_interval_count"],
        "primary complete-window count",
    )
    selected_resampled = np.zeros(query_resampled.shape[0], dtype=bool)
    for window in windows:
        selected_resampled |= (
            (query_resampled[:, 0] >= window["start_time_s"])
            & (query_resampled[:, 0] < window["end_time_s"])
        )
    positions = query_resampled[selected_resampled, 1:4]
    _same(positions.shape[0], len(windows) * 5, "selected query 1 Hz sample count")
    distances, _ = cKDTree(positions).query(map_trajectory[:, 1:4], k=1)
    selected_map = distances <= OVERLAP_CONTRACT["radius_m"]
    selected_query = np.zeros(query_trajectory.shape[0], dtype=bool)
    for window in windows:
        selected_query |= (
            (query_trajectory[:, 0] >= window["start_time_s"])
            & (query_trajectory[:, 0] < window["end_time_s"])
        )
    return {
        "complete_five_second_windows": windows,
        "map_selected_timestamp_us": [
            int(round(value))
            for value, selected in zip(map_values[:, 0], selected_map)
            if selected
        ],
        "query_selected_timestamp_us": [
            int(round(value))
            for value, selected in zip(query_values[:, 0], selected_query)
            if selected
        ],
        "selected_query_1hz_sample_count": int(positions.shape[0]),
        "selection_definition": {
            "map": "native map poses within 5 m in fixed ENU_ref of selected covered query 1 Hz samples",
            "query": "native query poses in complete half-open nonoverlapping 5 s windows carved from each covered island",
            "point_cloud_payload_consulted": False,
            "registration_result_consulted": False,
        },
    }


def _verify_pair_selection(
    *, runtime_root: Path, pair_contract: Mapping[str, Any], ranked: Sequence[Mapping[str, Any]],
    raw_results: Mapping[tuple[str, str], Mapping[str, Any]], trajectories: Mapping[str, np.ndarray],
    values_by_id: Mapping[str, np.ndarray], resampled: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str], Mapping[str, Any], Mapping[str, Any]]:
    if len(ranked) < 3:
        _fail("fewer than three eligible pairs are available")
    selected = [dict(row) for row in ranked[:3]]
    primary_key = (str(selected[0]["map_sequence_id"]), str(selected[0]["query_sequence_id"]))
    if primary_key[0] == primary_key[1]:
        _fail("primary pair uses the same sequence")
    raw = raw_results[primary_key]
    selection = _select_timestamps_independent(
        map_trajectory=trajectories[primary_key[0]],
        query_trajectory=trajectories[primary_key[1]],
        map_values=values_by_id[primary_key[0]],
        query_values=values_by_id[primary_key[1]],
        query_resampled=resampled[primary_key[1]],
        primary_raw=raw,
    )
    expected = {
        "PRIMARY_PAIR": selected[0],
        "RESERVE_PAIR_1": selected[1],
        "RESERVE_PAIR_2": selected[2],
        "eligible_pair_count": len(ranked),
        "primary_contiguous_covered_intervals": raw["contiguous_covered_intervals"],
        "primary_complete_five_second_windows": selection["complete_five_second_windows"],
        "ranking_rule": RANKING_RULE,
        "reserve_activation_policy": pair_contract["reserve_pair_activation"],
        "selection_inputs": "GT_ONLY_FIXED_ENU_REF",
    }
    _same(_load_json(runtime_root / "boreas_v2_pair_selection.json"), expected, "primary/reserve pair freeze")
    expected_md = (
        "# Boreas v2 frozen GT-only pairs\n\n"
        f"- PRIMARY: `{primary_key[0]} -> {primary_key[1]}`\n"
        f"- RESERVE 1: `{selected[1]['map_sequence_id']} -> {selected[1]['query_sequence_id']}`\n"
        f"- RESERVE 2: `{selected[2]['map_sequence_id']} -> {selected[2]['query_sequence_id']}`\n"
        f"- Eligible directed pairs: {len(ranked)} / 812\n\n"
        "Reserve activation is limited to the frozen infrastructure failures; registration or publication outcomes can never trigger a switch.\n"
    )
    _same((runtime_root / "boreas_v2_pair_selection.md").read_text(encoding="utf-8"), expected_md, "pair selection Markdown")
    return primary_key, selection, expected


def _metadata_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        source = {
            "key": row["key"],
            "last_modified": row["last_modified"],
            "size_bytes": row["size_bytes"],
        }
        digest.update(
            json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            + b"\n"
        )
    return digest.hexdigest()


def _verify_metadata_and_allowlist(
    *, boreas_v1_root: Path, runtime_root: Path, primary_key: tuple[str, str],
    selection: Mapping[str, Any], values_by_id: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    source_inventory = _load_json(boreas_v1_root / "boreas_remote_sequence_inventory.json")
    inventory_by_id = {
        str(row["sequence_id"]): row for row in source_inventory["boreas_original_sequences"]
    }
    raw_metadata = _csv_rows(
        runtime_root / "boreas_v2_primary_pair_lidar_metadata_inventory.csv", METADATA_FIELDS
    )
    metadata: list[dict[str, Any]] = []
    for row in raw_metadata:
        try:
            timestamp = int(row["timestamp_us"])
            size = int(row["size_bytes"])
        except ValueError as error:
            raise BoreasExternalV2Stage1VerificationError("invalid primary-pair metadata integer") from error
        if size < 0 or re.fullmatch(r"\d{16}", str(timestamp)) is None:
            _fail("invalid primary-pair metadata timestamp or size")
        sequence = row["sequence_id"]
        _same(row["key"], f"{sequence}/lidar/{timestamp}.bin", "metadata key/timestamp binding")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", row["last_modified"]) is None:
            _fail("invalid primary-pair metadata last-modified value")
        metadata.append({**row, "timestamp_us": timestamp, "size_bytes": size})

    expected_sequence_order = [
        sequence
        for sequence in primary_key
        for _ in range(values_by_id[sequence].shape[0])
    ]
    _same([row["sequence_id"] for row in metadata], expected_sequence_order, "metadata sequence/order closure")
    metadata_by_key: dict[str, dict[str, Any]] = {}
    listing_sha: dict[str, str] = {}
    for sequence in primary_key:
        rows = [row for row in metadata if row["sequence_id"] == sequence]
        timestamps = [row["timestamp_us"] for row in rows]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            _fail(f"metadata timestamps are not strictly ordered: {sequence}")
        expected_timestamps = [int(round(value)) for value in values_by_id[sequence][:, 0]]
        _same(timestamps, expected_timestamps, f"metadata/pose timestamp inventory {sequence}")
        source = inventory_by_id[sequence]
        _same(len(rows), source.get("lidar_object_count"), f"metadata object count {sequence}")
        _same(sum(row["size_bytes"] for row in rows), source.get("lidar_remote_bytes"), f"metadata bytes {sequence}")
        _same(timestamps[0], source.get("first_lidar_timestamp"), f"metadata first timestamp {sequence}")
        _same(timestamps[-1], source.get("last_lidar_timestamp"), f"metadata last timestamp {sequence}")
        listing_sha[sequence] = _metadata_digest(rows)
        _same(listing_sha[sequence], source.get("lidar_listing_rows_sha256"), f"authenticated metadata digest {sequence}")
        for row in rows:
            if row["key"] in metadata_by_key:
                _fail(f"duplicate primary metadata key: {row['key']}")
            metadata_by_key[row["key"]] = row

    allowlist_rows: list[dict[str, Any]] = []
    for role, sequence, timestamps, reason in (
        (
            "TARGET_MAP",
            primary_key[0],
            selection["map_selected_timestamp_us"],
            "native map pose within 5 m of selected covered query 1 Hz GT sample",
        ),
        (
            "QUERY",
            primary_key[1],
            selection["query_selected_timestamp_us"],
            "native query pose in frozen complete nonoverlapping covered 5 s window",
        ),
    ):
        for timestamp in timestamps:
            key = f"{sequence}/lidar/{timestamp}.bin"
            source = metadata_by_key.get(key)
            if source is None:
                _fail(f"independently selected key absent from authenticated listing: {key}")
            allowlist_rows.append(
                {
                    "selection_role": role,
                    **source,
                    "selection_reason": reason,
                }
            )
    role_order = {"TARGET_MAP": 0, "QUERY": 1}
    allowlist_rows.sort(key=lambda row: (role_order[row["selection_role"]], row["key"]))
    frozen_allowlist = _csv_rows(
        runtime_root / "boreas_v2_stage2_download_allowlist.csv", ALLOWLIST_FIELDS
    )
    _same(
        frozen_allowlist,
        [_string_row(row, ALLOWLIST_FIELDS) for row in allowlist_rows],
        "independently selected Stage-2 allowlist",
    )

    by_role = {
        role: [row for row in allowlist_rows if row["selection_role"] == role]
        for role in ("TARGET_MAP", "QUERY")
    }
    if any(not rows for rows in by_role.values()):
        _fail("Stage-2 allowlist has an empty role")
    role_summary = {
        role: {
            "first_timestamp_us": rows[0]["timestamp_us"],
            "last_timestamp_us": rows[-1]["timestamp_us"],
            "object_count": len(rows),
            "remote_bytes": sum(row["size_bytes"] for row in rows),
            "sequence_id": rows[0]["sequence_id"],
        }
        for role, rows in by_role.items()
    }
    full_sequence = {
        sequence: {
            "first_timestamp_us": rows[0]["timestamp_us"],
            "last_timestamp_us": rows[-1]["timestamp_us"],
            "object_count": len(rows),
            "remote_bytes": sum(row["size_bytes"] for row in rows),
            "listing_sha256": listing_sha[sequence],
        }
        for sequence in primary_key
        for rows in [[row for row in metadata if row["sequence_id"] == sequence]]
    }
    selected_bytes = sum(row["size_bytes"] for row in allowlist_rows)
    expected_plan = {
        "allowlist_file": "boreas_v2_stage2_download_allowlist.csv",
        "allowlist_sha256": sha256_file(runtime_root / "boreas_v2_stage2_download_allowlist.csv"),
        "downloaded_lidar_bytes": 0,
        "downloaded_lidar_object_count": 0,
        "estimated_download_bytes": selected_bytes,
        "estimated_download_GB_decimal": selected_bytes / 1_000_000_000,
        "estimated_download_GiB": selected_bytes / (1024**3),
        "full_primary_sequence_metadata": full_sequence,
        "metadata_listing_invocations": [
            {
                "command_class": "aws s3 ls --recursive --no-sign-request",
                "listing_sha256": listing_sha[sequence],
                "payload_download": False,
                "prefix": f"s3://boreas/{sequence}/lidar/",
                "sequence_id": sequence,
            }
            for sequence in primary_key
        ],
        "metadata_only": True,
        "primary_pair": {
            "map_sequence_id": primary_key[0],
            "query_sequence_id": primary_key[1],
        },
        "selection": selection["selection_definition"],
        "selected_complete_five_second_window_count": len(selection["complete_five_second_windows"]),
        "selected_objects": role_summary,
        "stage2_execution_authorized": False,
    }
    plan = _load_json(runtime_root / "boreas_v2_stage2_download_plan.json")
    _same(plan, expected_plan, "Stage-2 metadata-only download plan")
    return allowlist_rows, role_summary, plan


def _verify_disk_budget(
    *, runtime_root: Path, role_summary: Mapping[str, Mapping[str, Any]], plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    selected = int(plan["estimated_download_bytes"])
    map_bytes = int(role_summary["TARGET_MAP"]["remote_bytes"])
    query_bytes = int(role_summary["QUERY"]["remote_bytes"])
    _same(selected, map_bytes + query_bytes, "selected-byte role closure")
    decoded = selected * 2
    target_map = map_bytes * 2
    canonical = selected * 2
    temporary = selected
    subtotal = selected + decoded + target_map + canonical + temporary
    safety = math.ceil(subtotal * 1.5)
    actual = _load_json(runtime_root / "boreas_v2_stage2_disk_budget.json")
    current_free = actual.get("current_free_bytes")
    if isinstance(current_free, bool) or not isinstance(current_free, int) or current_free < 0:
        _fail("recorded disk free space is invalid")
    expected = {
        "canonical_bundle_bytes": canonical,
        "current_free_bytes": current_free,
        "current_machine_meets_safety_budget": current_free >= safety,
        "decoded_or_unpacked_working_bytes": decoded,
        "download_bytes": selected,
        "formula": "1.5 * (download + decoded_working + target_map + canonical_bundle + temporary_processing)",
        "minimum_safety_multiplier": 1.5,
        "query_download_bytes": query_bytes,
        "required_safe_disk_GB_decimal": safety / 1_000_000_000,
        "required_safe_disk_GiB": safety / (1024**3),
        "required_safe_disk_bytes": safety,
        "subtotal_before_safety_bytes": subtotal,
        "target_map_bytes": target_map,
        "target_map_download_bytes": map_bytes,
        "temporary_processing_bytes": temporary,
    }
    _same(actual, expected, "Stage-2 disk budget")
    if safety < math.ceil(subtotal * 1.5):
        _fail("disk safety multiplier is below 1.5")
    return actual


def _live_lidar_payloads(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("*.bin")
        if path.is_file() and path.parent.name.lower() == "lidar"
    )


def _verify_zero_execution(*, data_root: Path, runtime_root: Path) -> None:
    lidar = _live_lidar_payloads(data_root)
    if lidar:
        _fail(f"LiDAR payload exists in Stage-1 data root: {lidar[0]}")
    no_lidar = _load_json(runtime_root / "NO_LIDAR_PAYLOAD_ATTESTATION.json")
    expected_lidar = {
        "data_root": str(data_root),
        "downloaded_lidar_bytes": 0,
        "downloaded_lidar_object_count": 0,
        "downloaded_lidar_payload_count": 0,
        "initial_local_lidar_bin_count": 0,
        "local_lidar_bin_count": 0,
        "local_lidar_bin_paths": [],
        "metadata_listing_only": True,
        "pass": True,
        "payload_materialization_authorized": False,
        "status": "PASS",
    }
    _same(no_lidar, expected_lidar, "NO_LIDAR_PAYLOAD attestation")

    no_icp = _load_json(runtime_root / "NO_ICP_ATTESTATION.json")
    for key in (
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
        "structured_result_scan_error_count",
    ):
        _same(no_icp.get(key), 0, f"NO_ICP {key}")
    for key in (
        "estimated_transform_evidence",
        "estimated_transform_files",
        "structured_result_scan_error_files",
    ):
        _same(no_icp.get(key), [], f"NO_ICP {key}")
    _same(no_icp.get("pass"), True, "NO_ICP pass")
    _same(no_icp.get("status"), "PASS", "NO_ICP status")
    source_audit = no_icp.get("static_source_audit", {})
    _same(source_audit.get("pass"), True, "preparation source audit")
    _same(source_audit.get("violations"), [], "preparation source violations")
    if not isinstance(source_audit.get("python_file_count"), int) or source_audit["python_file_count"] <= 0:
        _fail("preparation source audit file count is invalid")


def _verify_test_baseline(root: Path) -> Mapping[str, Any]:
    baseline = _load_json(root / "test_baseline_status.json")
    _same(baseline.get("SOURCE_ONLY_TEST_BASELINE"), "PASS", "source-only baseline status")
    _same(
        baseline.get("EXTERNAL_QUALIFICATION_BASELINE"),
        "UNAVAILABLE",
        "external qualification baseline status",
    )
    _same(
        baseline.get("initial_baseline"),
        {
            "collected": 771,
            "command": "python -m pytest -q",
            "failed": 1,
            "passed": 761,
            "skipped": 9,
        },
        "initial test baseline",
    )
    source = baseline.get("source_only_baseline", {})
    for key, expected in (("status", "PASS"), ("errors", 0), ("failed", 0)):
        _same(source.get(key), expected, f"source-only baseline {key}")
    for key in ("collected", "passed", "skipped"):
        if isinstance(source.get(key), bool) or not isinstance(source.get(key), int) or source[key] < 0:
            _fail(f"source-only baseline {key} is invalid")
    _same(source["collected"], source["passed"] + source["skipped"], "source-only test count closure")
    _same(
        source.get("command"),
        "PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX=/home/lj/.local/share/degen-lio-micromamba python -m pytest -q",
        "source-only baseline command",
    )
    expected_skip_reasons = [
        {"count": 2, "reason": "source-only ZIP excludes the historical Git baseline object"},
        {"count": 1, "reason": "source-only ZIP excludes the historical Phase B tag object"},
        {"count": 1, "reason": "source-only ZIP excludes the historical formal Phase A results"},
        {"count": 1, "reason": "source-only ZIP excludes the historical Phase B raw results"},
        {
            "count": 1,
            "reason": (
                f"{PCL_EXTERNAL_BUNDLE}: source-only package: external historical "
                "qualification bundle unavailable"
            ),
        },
        {"count": 1, "reason": "source-only ZIP excludes the historical v1 failure bundle"},
        {"count": 3, "reason": "source-only ZIP excludes the historical v2 Git object"},
    ]
    _same(source.get("skip_reasons"), expected_skip_reasons, "source-only skip reasons")
    _same(sum(row["count"] for row in expected_skip_reasons), source["skipped"], "skip-reason count closure")
    external = baseline.get("external_qualification_baseline", {})
    expected_external = {
        "command": (
            "ZPRM_REQUIRE_PCL_V3_EXTERNAL_QUALIFICATION=1 python -m pytest -q "
            "tests/test_runtime_lifecycle_qualification.py::"
            "test_fixed_pcl_v3_input_inventory_matches_contract"
        ),
        "errors": 0,
        "exit_code": 1,
        "failed": 1,
        "missing_path": str(PCL_EXTERNAL_BUNDLE),
        "passed": 0,
        "reason": (
            f"{PCL_EXTERNAL_BUNDLE}: source-only package: external historical "
            "qualification bundle unavailable"
        ),
        "required_environment": "ZPRM_REQUIRE_PCL_V3_EXTERNAL_QUALIFICATION=1",
        "status": "UNAVAILABLE",
    }
    _same(external, expected_external, "strict external qualification baseline")
    expected_md = (
        "# Public-data v2 test baselines\n\n"
        "- `SOURCE_ONLY_TEST_BASELINE=PASS`: "
        f"{source['collected']} collected, {source['passed']} passed, "
        f"{source['skipped']} skipped, 0 failed, 0 errors.\n"
        "- `EXTERNAL_QUALIFICATION_BASELINE=UNAVAILABLE`: strict mode fails because "
        f"`{PCL_EXTERNAL_BUNDLE}` is unavailable; it is not reported as PASS.\n"
    )
    _same((root / "test_baseline_status.md").read_text(encoding="utf-8"), expected_md, "test baseline Markdown")
    return baseline


EXPECTED_ELIGIBILITY = {
    "BOREAS_EXTERNAL_V2_STAGE1_READY": True,
    "E01": "PASS",
    "E02": "PASS_WITH_DOCUMENTED_LIMITATION",
    "E03": "PASS_PREREGISTERED_DISTINCT_TARGET_MAP",
    "E04": "PASS",
    "E05": "PASS_WITH_DOCUMENTED_LIMITATION",
    "E06": "FROZEN_FOR_FUTURE_STAGE2_CURRENT_COUNTS_ZERO",
    "E07": "PASS_PREREGISTERED",
    "E08": "PASS_PREREGISTERED",
    "E09": "PASS",
    "E10": "PASS_WITH_UNKNOWN_COMPONENTS_PRESERVED",
    "E11": "PASS_SUPPLEMENTARY_BOUNDARY",
    "E12": "PASS_PREREGISTERED",
    "E13": "PASS_PREREGISTERED",
    "E14": "PASS_PAIR_FREEZE",
    "GT_ONLY_OVERLAP": "PASS",
    "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
    "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD": True,
    "REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED": False,
    "REAL_REGISTRATION_AUTHORIZED": False,
    "actual_trials": 0,
    "downloaded_lidar_bytes": 0,
    "downloaded_lidar_object_count": 0,
    "downloaded_lidar_payload_count": 0,
    "planned_trials": 0,
    "real_trial_result_count": 0,
    "registration_execution_count": 0,
    "rich_snapshot_count": 0,
    "snapshot_count": 0,
    "weak_snapshot_count": 0,
}


def _verify_decision_and_summary(
    *, root: Path, pair_selection: Mapping[str, Any], baseline: Mapping[str, Any], plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    eligibility = _load_json(root / "public_data_v2_stage1_eligibility.json")
    _same(eligibility, EXPECTED_ELIGIBILITY, "v2 Stage-1 eligibility")
    summary = _load_json(root / "public_data_v2_stage1_summary.json")
    _same(set(summary), set(EXPECTED_ELIGIBILITY) | {"answers", "final_conclusion", "primary_pair", "reserve_pairs"}, "summary field closure")
    for key, expected in EXPECTED_ELIGIBILITY.items():
        _same(summary.get(key), expected, f"summary {key}")
    answers = summary.get("answers")
    if not isinstance(answers, list) or len(answers) != 26 or any(not isinstance(item, str) or not item for item in answers):
        _fail("Chinese Stage-1 summary must contain exactly 26 nonempty answers")
    _same(summary.get("primary_pair"), pair_selection["PRIMARY_PAIR"], "summary primary pair")
    _same(
        summary.get("reserve_pairs"),
        [pair_selection["RESERVE_PAIR_1"], pair_selection["RESERVE_PAIR_2"]],
        "summary reserve pairs",
    )
    conclusion = str(summary.get("final_conclusion", ""))
    for phrase in (
        "BOREAS_EXTERNAL_V2_STAGE1_READY=true",
        "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD=true",
        "supplementary external-validation",
        "LiDAR-assisted provenance",
        "本任务未执行 ICP",
    ):
        if phrase not in conclusion:
            _fail(f"summary conclusion boundary missing: {phrase}")
    forbidden_true = (
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=true",
        "REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED=true",
        "PUBLIC_DATA_V2_RUN_AUTHORIZED=true",
        "REAL_REGISTRATION_AUTHORIZED=true",
    )
    if any(token in conclusion for token in forbidden_true):
        _fail("summary conclusion contains a forbidden authorization")
    _same(baseline["source_only_baseline"]["failed"], 0, "summary source-only failure count")
    _same(plan.get("downloaded_lidar_object_count"), 0, "summary plan downloaded object count")
    expected_md_lines = [
        "# Boreas External Validation v2 Stage-1 总结",
        "",
        "**BOREAS_EXTERNAL_V2_STAGE1_READY=true**",
        "**READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD=true**",
        "",
    ]
    expected_md_lines.extend(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
    expected_md_lines.extend(
        [
            "",
            "## 冻结零计数与授权边界",
            "",
            "```text",
            "weak_snapshot_count=0",
            "rich_snapshot_count=0",
            "snapshot_count=0",
            "planned_trials=0",
            "actual_trials=0",
            "registration_execution_count=0",
            "downloaded_lidar_payload_count=0",
            "PUBLIC_DATA_V2_RUN_AUTHORIZED=false",
            "REAL_DATA_MAIN_EXPERIMENT_AUTHORIZED=false",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false",
            "```",
            "",
            "## 最终结论",
            "",
            conclusion,
            "",
        ]
    )
    _same(
        (root / "public_data_v2_stage1_summary.md").read_text(encoding="utf-8"),
        "\n".join(expected_md_lines),
        "summary Markdown/JSON projection",
    )
    return summary


def verify_boreas_external_v2_stage1(
    *, repository: str | Path, data_root: str | Path, runtime_root: str | Path
) -> dict[str, Any]:
    """Authenticate and independently recompute the complete v2 Stage-1 closure."""

    repository_input = Path(repository)
    data_input = Path(data_root)
    runtime_input = Path(runtime_root)
    if any(path.is_symlink() for path in (repository_input, data_input, runtime_input)):
        _fail("repository, data, and runtime roots must not be symlinks")
    repository_path = repository_input.resolve(strict=True)
    data_path = data_input.resolve(strict=True)
    root = runtime_input.resolve(strict=True)
    if not repository_path.is_dir() or not data_path.is_dir() or not root.is_dir():
        _fail("repository, data, and runtime roots must be directories")

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository_path, text=True
    ).strip()
    if branch not in {EXPECTED_BRANCH, STAGE2_PREPARATION_BRANCH}:
        _fail(
            "verification branch differs: expected frozen Stage-1 branch or "
            f"{STAGE2_PREPARATION_BRANCH!r}, got {branch!r}"
        )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BOREAS_V1_COMMIT, "HEAD"],
        cwd=repository_path,
    ).returncode:
        _fail("Boreas v1 commit is not an ancestor of current HEAD")

    manifest = _verify_runtime_closure(root)
    producer = str(manifest["producer_commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", producer, "HEAD"], cwd=repository_path
    ).returncode:
        _fail("v2 producer commit is not an ancestor of current HEAD")

    v1_closure_root = repository_path / "frozen_assets/public_data_validation_v1_closure"
    closure_report = verify_public_data_v1_closure(
        repository=repository_path, closure_root=v1_closure_root
    )
    _same(
        manifest.get("v1_closure_verification"),
        closure_report,
        "manifest v1 closure verification",
    )
    for name in V1_CLOSURE_COPIES:
        _same(
            (root / name).read_bytes(),
            (v1_closure_root / name).read_bytes(),
            f"copied immutable v1 closure {name}",
        )

    protocol_state = _verify_protocols(repository_path, root, manifest)
    boreas_v1_root = repository_path / "frozen_assets/real_data_boreas_stage1_v1"
    _verify_boreas_v1_bundle(boreas_v1_root)
    evidence_paths, _, _ = _authenticate_evidence(
        data_root=data_path, boreas_v1_root=boreas_v1_root, runtime_root=root
    )
    _verify_requalification(boreas_v1_root=boreas_v1_root, runtime_root=root)
    reports, trajectories, values_by_id, calibration_sha = _load_eligible_trajectories(
        boreas_v1_root=boreas_v1_root, evidence_paths=evidence_paths
    )
    ranked, raw_results, resampled = _verify_overlap(
        runtime_root=root,
        reports=reports,
        trajectories=trajectories,
        calibration_sha=calibration_sha,
    )
    primary_key, selection, pair_selection = _verify_pair_selection(
        runtime_root=root,
        pair_contract=protocol_state["pair_contract"],
        ranked=ranked,
        raw_results=raw_results,
        trajectories=trajectories,
        values_by_id=values_by_id,
        resampled=resampled,
    )
    allowlist, role_summary, plan = _verify_metadata_and_allowlist(
        boreas_v1_root=boreas_v1_root,
        runtime_root=root,
        primary_key=primary_key,
        selection=selection,
        values_by_id=values_by_id,
    )
    _verify_disk_budget(runtime_root=root, role_summary=role_summary, plan=plan)
    _verify_zero_execution(data_root=data_path, runtime_root=root)
    baseline = _verify_test_baseline(root)
    summary = _verify_decision_and_summary(
        root=root, pair_selection=pair_selection, baseline=baseline, plan=plan
    )
    return {
        "BOREAS_EXTERNAL_V2_STAGE1_READY": True,
        "BOREAS_EXTERNAL_V2_STAGE1_VERIFICATION_PASS": True,
        "READY_FOR_SEPARATE_STAGE2_LIDAR_DOWNLOAD": True,
        "data_evidence_file_count": len(evidence_paths),
        "downloaded_lidar_payload_count": 0,
        "eligible_directed_pair_count": len(ranked),
        "eligible_reference_sequence_count": len(reports),
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "primary_pair": {
            "map_sequence_id": primary_key[0],
            "query_sequence_id": primary_key[1],
        },
        "registration_execution_count": 0,
        "selected_stage2_lidar_object_count": len(allowlist),
        "summary": summary,
        "verification_pass": True,
    }


__all__ = [
    "BoreasExternalV2Stage1VerificationError",
    "verify_boreas_external_v2_stage1",
]
