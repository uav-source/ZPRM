"""Independent verifier for the Boreas v2 Stage-2 storage-plan closure.

This verifier deliberately does not import the storage-plan producer or the
storage planner.  Frozen Stage-1 identities, scientific constants, lifecycle
rules, and all budget arithmetic are redeclared here so a self-consistent
producer defect (or a re-signed semantic tamper) cannot verify itself.

The verifier is metadata-only: it never downloads a Boreas object, decodes a
real point cloud, builds a real map, or invokes an Open3D/PCL backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


GIB = 1024**3

STAGE1_RELATIVE_ROOT = Path(
    "frozen_assets/public_data_external_validation_v2_boreas_stage1"
)
EXPECTED_BRANCH = "prep/boreas-v2-stage2-storage-optimization"
STAGE1_TAG_COMMIT = "fb6d84fc41a252fc1934716f96473fad1634ebdc"
PROTECTED_TAGS = {
    "archive/public-data-v1-screening-closed": (
        "59a1fa000d1de7fc4df2b7feeb5b4fb2f617750c"
    ),
    "preparation/public-data-external-v2-boreas-stage1": STAGE1_TAG_COMMIT,
    "execution/synthetic-confirmatory-v3-requalified": (
        "01b19b1ba89376cb53aa082b77032b562888eb7b"
    ),
}
STAGE1_FROZEN_MANIFEST_FILE_SHA256 = (
    "71cfd78aa588c67dd28f8f8be87b7514c20ed090e75a8433583362253a97a8be"
)
STAGE1_MANIFEST_ROOT_SHA256 = (
    "677346ded7d4d45be1877dc19d0c316e80e87699784a441ed1d8b69848eae44f"
)
STAGE1_SHA256SUMS_SHA256 = (
    "cbda458050de5bd3571d18503f7e7db54277e22e299b61116f5a9801df049967"
)
STAGE1_PAIR_SELECTION_SHA256 = (
    "b28a52498a2fddc1b80d43c626b75179358066677ce8f493be22afba7a19639c"
)
STAGE1_ALLOWLIST_SHA256 = (
    "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
)
STAGE1_DOWNLOAD_PLAN_SHA256 = (
    "3687a4e53945a97fd88e7f32ee2f57f5a0ddb0cac1e24ad452d047df218caa0b"
)
STAGE1_OLD_BUDGET_SHA256 = (
    "f8179dbc672a3ac08ada50f2835f86fec5e4a6b651d6d5cca6a61fc4fbd991a5"
)
STAGE1_NO_ICP_SHA256 = (
    "3296df5556ca6e6f0a6cf55f30dbdda0d9a776e77f23a960e8ab502c2ba52876"
)
STAGE1_NO_LIDAR_SHA256 = (
    "1765fab5fe25217f18594411efde0adbb5d05edfb31fee213c08b49f84deb8ac"
)
BACKEND_PARAMETER_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)

PRIMARY_MAP_SEQUENCE = "boreas-2021-11-14-09-47"
PRIMARY_QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
EXPECTED_ALLOWLIST_OBJECT_COUNT = 20_061
EXPECTED_REMOTE_BYTES = 104_158_637_472
EXPECTED_MAP_OBJECT_COUNT = 8_202
EXPECTED_MAP_REMOTE_BYTES = 41_998_817_280
EXPECTED_QUERY_OBJECT_COUNT = 11_859
EXPECTED_QUERY_REMOTE_BYTES = 62_159_820_192
EXPECTED_SNAPSHOT_COUNT = 100
EXPECTED_WEAK_SNAPSHOT_COUNT = 50
EXPECTED_RICH_SNAPSHOT_COUNT = 50
EXPECTED_BACKEND_TRIAL_COUNT = 200
PREPROCESSING_UNRESOLVED = (
    "PREPROCESSING_PARAMETER_REQUIRES_STAGE2_PREREGISTRATION"
)

PAYLOAD_FILES = frozenset(
    {
        "NO_ICP_ATTESTATION.json",
        "NO_LIDAR_PAYLOAD_ATTESTATION.json",
        "boreas_v2_stage2_disk_budget_optimized.csv",
        "boreas_v2_stage2_disk_budget_optimized.json",
        "boreas_v2_stage2_execution_modes.json",
        "boreas_v2_stage2_storage_readiness.json",
        "canonical_bundle_storage_contract_v2.json",
        "source_only_pytest_junit.xml",
        "source_only_test_status.json",
        "stage2_checkpoint_contract.json",
        "stage2_object_lifecycle_contract.csv",
        "stage2_object_lifecycle_contract.json",
        "stage2_scientific_lock_contract.json",
        "stage2_storage_architecture_contract.json",
        "stage2_storage_budget_original_breakdown.csv",
        "stage2_storage_budget_original_breakdown.json",
        "stage2_storage_budget_original_explanation.md",
        "stage2_storage_summary.json",
        "stage2_storage_summary.md",
        "stage2_streaming_map_contract.json",
        "stage2_streaming_query_contract.json",
        "stage2_temp_file_contract.json",
    }
)
REQUIRED_FILES = PAYLOAD_FILES | {"frozen_manifest.json", "SHA256SUMS"}

ORIGINAL_BREAKDOWN_FIELDS = (
    "component",
    "unit_count",
    "bytes_per_unit",
    "raw_bytes",
    "duplication_factor",
    "lifetime",
    "simultaneously_live",
    "peak_contribution_bytes",
    "retained_after_stage2",
    "reason",
    "source_of_estimate",
)
LIFECYCLE_FIELDS = (
    "object_class",
    "lifecycle_class",
    "creation_or_source",
    "consumed_by",
    "deletion_or_retention_rule",
    "persistent",
    "scientific_evidence",
    "resume_rule",
)
BUDGET_CSV_FIELDS = (
    "mode",
    "persistent_bytes",
    "persistent_GiB",
    "temporary_peak_bytes",
    "temporary_peak_GiB",
    "simultaneously_live_peak_bytes",
    "simultaneously_live_peak_GiB",
    "safety_margin_fraction",
    "safety_margin_bytes",
    "safety_margin_GiB",
    "recommended_free_disk_bytes",
    "recommended_free_disk_GiB",
    "current_free_bytes",
    "CURRENT_DISK_SUFFICIENT",
    "additional_bytes_required",
    "additional_GiB_required",
    "minimum_free_disk_required_before_start_bytes",
    "abort_if_free_disk_below_bytes",
    "runtime_low_disk_watermark_bytes",
    "watermark_action",
)

AUTHORIZATION_FIELDS = (
    "STAGE2_DOWNLOAD_AUTHORIZED",
    "PUBLIC_DATA_V2_RUN_AUTHORIZED",
    "REAL_REGISTRATION_AUTHORIZED",
    "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
)
ZERO_EXECUTION_FIELDS = (
    "actual_trials",
    "downloaded_lidar_bytes",
    "downloaded_lidar_object_count",
    "downloaded_lidar_payload_count",
    "planned_trials",
    "real_trial_result_count",
    "registration_execution_count",
    "rich_snapshot_count",
    "snapshot_count",
    "weak_snapshot_count",
)


class BoreasStage2StorageVerificationError(RuntimeError):
    """The Stage-2 storage closure or its declared meaning was altered."""


def _fail(message: str) -> None:
    raise BoreasStage2StorageVerificationError(message)


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: {actual!r} != {expected!r}")


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    """Require a closed JSON-object schema, not merely a compatible subset.

    The frozen closure is an authorization boundary.  Unknown fields are not
    harmless extensions because a re-signed field can contradict the fields
    the verifier understands (for example, by claiming a download is
    authorized next to the verified false authorization).  Every contract
    object therefore fails closed on additions as well as omissions.
    """

    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    _same(set(value), expected, f"{label} field set")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, canonical: bool = True) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoreasStage2StorageVerificationError(f"invalid JSON: {path}") from error
    if canonical and path.read_bytes() != _canonical_json_bytes(value):
        _fail(f"noncanonical JSON: {path}")
    return value


def _safe_basename(name: str) -> str:
    path = PurePosixPath(name)
    if len(path.parts) != 1 or path.is_absolute() or name in {"", ".", ".."}:
        _fail(f"unsafe closure path: {name!r}")
    return name


def _parse_sums(root: Path, expected_files: frozenset[str]) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        _fail("SHA256SUMS is absent or unsafe")
    rows: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail(f"malformed SHA256SUMS row: {line!r}")
        digest, raw_name = match.groups()
        name = _safe_basename(raw_name)
        if name in rows or name == "SHA256SUMS":
            _fail(f"duplicate or recursive SHA256SUMS row: {name}")
        path = root / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            _fail(f"SHA256 closure mismatch: {name}")
        rows[name] = digest
    _same(set(rows), set(expected_files) - {"SHA256SUMS"}, "SHA256SUMS inventory")
    entries = list(root.iterdir())
    _same({path.name for path in entries}, set(expected_files), "exact closure inventory")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        _fail("non-regular entry in immutable storage closure")
    return rows


def _csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _same(reader.fieldnames, list(fields), f"CSV header {path.name}")
        return list(reader)


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _verify_csv_projection(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    actual = _csv_rows(path, fields)
    expected = [
        {field: _csv_value(row.get(field)) for field in fields}
        for row in rows
    ]
    _same(actual, expected, f"CSV/JSON projection {path.name}")


def _require_zero_contract(value: Mapping[str, Any], label: str) -> None:
    for field in AUTHORIZATION_FIELDS:
        _same(value.get(field), False, f"{label} {field}")
    for field in ZERO_EXECUTION_FIELDS:
        _same(value.get(field), 0, f"{label} {field}")


def _verify_stage1(repository: Path) -> dict[str, Any]:
    root = repository / STAGE1_RELATIVE_ROOT
    if root.is_symlink() or not root.is_dir():
        _fail("frozen Stage-1 root is absent or unsafe")
    expected_hashes = {
        "frozen_manifest.json": STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "SHA256SUMS": STAGE1_SHA256SUMS_SHA256,
        "boreas_v2_pair_selection.json": STAGE1_PAIR_SELECTION_SHA256,
        "boreas_v2_stage2_download_allowlist.csv": STAGE1_ALLOWLIST_SHA256,
        "boreas_v2_stage2_download_plan.json": STAGE1_DOWNLOAD_PLAN_SHA256,
        "boreas_v2_stage2_disk_budget.json": STAGE1_OLD_BUDGET_SHA256,
        "NO_ICP_ATTESTATION.json": STAGE1_NO_ICP_SHA256,
        "NO_LIDAR_PAYLOAD_ATTESTATION.json": STAGE1_NO_LIDAR_SHA256,
    }
    for name, digest in expected_hashes.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            _fail(f"frozen Stage-1 file absent or unsafe: {name}")
        _same(_sha256_file(path), digest, f"frozen Stage-1 SHA {name}")

    manifest = _load_json(root / "frozen_manifest.json")
    _same(
        manifest.get("manifest_root_sha256"),
        STAGE1_MANIFEST_ROOT_SHA256,
        "frozen Stage-1 manifest root",
    )
    unsigned = dict(manifest)
    unsigned.pop("manifest_root_sha256", None)
    _same(_compact_sha256(unsigned), STAGE1_MANIFEST_ROOT_SHA256, "Stage-1 manifest self-hash")

    pair = _load_json(root / "boreas_v2_pair_selection.json")
    primary = pair.get("PRIMARY_PAIR")
    if not isinstance(primary, Mapping):
        _fail("Stage-1 primary pair is absent")
    _same(
        (primary.get("map_sequence_id"), primary.get("query_sequence_id")),
        (PRIMARY_MAP_SEQUENCE, PRIMARY_QUERY_SEQUENCE),
        "Stage-1 primary pair",
    )

    counts = {"TARGET_MAP": 0, "QUERY": 0}
    totals = {"TARGET_MAP": 0, "QUERY": 0}
    maxima = {"TARGET_MAP": 0, "QUERY": 0}
    seen: set[str] = set()
    allowlist = root / "boreas_v2_stage2_download_allowlist.csv"
    with allowlist.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _same(
            reader.fieldnames,
            [
                "selection_role",
                "sequence_id",
                "key",
                "timestamp_us",
                "last_modified",
                "size_bytes",
                "selection_reason",
            ],
            "Stage-1 allowlist header",
        )
        for row in reader:
            role = row["selection_role"]
            if role not in counts:
                _fail(f"unknown Stage-1 allowlist role: {role}")
            sequence = (
                PRIMARY_MAP_SEQUENCE if role == "TARGET_MAP" else PRIMARY_QUERY_SEQUENCE
            )
            key = row["key"]
            if (
                row["sequence_id"] != sequence
                or re.fullmatch(rf"{re.escape(sequence)}/lidar/[0-9]+\.bin", key) is None
                or key in seen
            ):
                _fail("Stage-1 allowlist pair/key identity changed")
            try:
                size = int(row["size_bytes"])
            except ValueError as error:
                raise BoreasStage2StorageVerificationError("invalid allowlist size") from error
            if size <= 0:
                _fail("nonpositive Stage-1 allowlist size")
            seen.add(key)
            counts[role] += 1
            totals[role] += size
            maxima[role] = max(maxima[role], size)
    _same(counts, {"TARGET_MAP": 8_202, "QUERY": 11_859}, "Stage-1 role counts")
    _same(
        totals,
        {"TARGET_MAP": 41_998_817_280, "QUERY": 62_159_820_192},
        "Stage-1 role bytes",
    )
    _same(sum(counts.values()), EXPECTED_ALLOWLIST_OBJECT_COUNT, "Stage-1 object count")
    _same(sum(totals.values()), EXPECTED_REMOTE_BYTES, "Stage-1 remote bytes")

    plan = _load_json(root / "boreas_v2_stage2_download_plan.json")
    _same(plan.get("estimated_download_bytes"), EXPECTED_REMOTE_BYTES, "Stage-1 plan bytes")
    selected = plan.get("selected_objects")
    if not isinstance(selected, Mapping):
        _fail("Stage-1 selected_objects missing")
    for role in ("TARGET_MAP", "QUERY"):
        selected_role = selected.get(role)
        if not isinstance(selected_role, Mapping):
            _fail(f"Stage-1 plan role absent: {role}")
        _same(selected_role.get("object_count"), counts[role], f"Stage-1 plan {role} count")
        _same(selected_role.get("remote_bytes"), totals[role], f"Stage-1 plan {role} bytes")

    old = _load_json(root / "boreas_v2_stage2_disk_budget.json")
    _same(old.get("download_bytes"), EXPECTED_REMOTE_BYTES, "legacy download bytes")
    _same(old.get("subtotal_before_safety_bytes"), 708_949_459_392, "legacy subtotal")
    _same(old.get("required_safe_disk_bytes"), 1_063_424_189_088, "legacy total")

    closure_names = (
        "frozen_manifest.json",
        "SHA256SUMS",
        "boreas_v2_pair_selection.json",
        "boreas_v2_stage2_download_allowlist.csv",
        "boreas_v2_stage2_download_plan.json",
        "boreas_v2_stage2_disk_budget.json",
    )
    return {
        "allowlist_bytes": allowlist.stat().st_size,
        "closure_bytes": sum((root / name).stat().st_size for name in closure_names),
        "map_max_object_bytes": maxima["TARGET_MAP"],
        "max_object_bytes": max(maxima.values()),
        "query_max_object_bytes": maxima["QUERY"],
    }


def _verify_manifest(root: Path) -> Mapping[str, Any]:
    _parse_sums(root, REQUIRED_FILES)
    manifest = _load_json(root / "frozen_manifest.json")
    if not isinstance(manifest, Mapping):
        _fail("storage manifest is not an object")
    _exact_keys(
        manifest,
        {
            "actual_execution_count",
            "allowlist_object_count",
            "allowlist_remote_bytes",
            "allowlist_sha256",
            "architecture_only",
            "downloaded_lidar_payload_count",
            "manifest_root_sha256",
            "payload",
            "producer_commit",
            "registration_execution_count",
            "schema_version",
            "stage1_frozen_manifest_file_sha256",
            "stage1_manifest_root_sha256",
            "stage1_tag_commit",
        },
        "storage manifest",
    )
    unsigned = dict(manifest)
    recorded_root = unsigned.pop("manifest_root_sha256", None)
    _same(recorded_root, _compact_sha256(unsigned), "storage manifest self-hash")
    _same(
        manifest.get("schema_version"),
        "boreas_v2_stage2_storage_optimization_manifest_v1",
        "storage manifest schema",
    )
    payload = manifest.get("payload")
    if not isinstance(payload, list):
        _fail("storage manifest payload is not a list")
    if any(
        not isinstance(row, Mapping)
        or set(row) != {"path", "sha256", "size_bytes"}
        or not isinstance(row.get("path"), str)
        for row in payload
    ):
        _fail("malformed manifest payload row")
    _same([row.get("path") for row in payload], sorted(PAYLOAD_FILES), "manifest payload order")
    for row in payload:
        name = _safe_basename(str(row["path"]))
        path = root / name
        _same(row.get("sha256"), _sha256_file(path), f"manifest payload SHA {name}")
        _same(row.get("size_bytes"), path.stat().st_size, f"manifest payload size {name}")
    for key, expected in (
        ("actual_execution_count", 0),
        ("allowlist_object_count", EXPECTED_ALLOWLIST_OBJECT_COUNT),
        ("allowlist_remote_bytes", EXPECTED_REMOTE_BYTES),
        ("allowlist_sha256", STAGE1_ALLOWLIST_SHA256),
        ("architecture_only", True),
        ("downloaded_lidar_payload_count", 0),
        ("registration_execution_count", 0),
        ("stage1_frozen_manifest_file_sha256", STAGE1_FROZEN_MANIFEST_FILE_SHA256),
        ("stage1_manifest_root_sha256", STAGE1_MANIFEST_ROOT_SHA256),
        ("stage1_tag_commit", STAGE1_TAG_COMMIT),
    ):
        _same(manifest.get(key), expected, f"storage manifest {key}")
    producer = manifest.get("producer_commit")
    if not isinstance(producer, str) or re.fullmatch(r"[0-9a-f]{40}", producer) is None:
        _fail("invalid storage producer commit")
    return manifest


def _verify_original_budget(root: Path) -> None:
    value = _load_json(root / "stage2_storage_budget_original_breakdown.json")
    _exact_keys(
        value,
        {
            "breakdown_closes_to_required_safe_bytes",
            "legacy_formula",
            "legacy_required_safe_disk_GiB",
            "legacy_required_safe_disk_bytes",
            "legacy_subtotal_before_safety_bytes",
            "old_budget_sha256",
            "rows",
            "schema_version",
        },
        "original budget contract",
    )
    _same(
        value.get("schema_version"),
        "boreas_v2_stage2_original_budget_breakdown_v1",
        "original budget schema",
    )
    _same(value.get("old_budget_sha256"), STAGE1_OLD_BUDGET_SHA256, "old budget binding")
    _same(value.get("legacy_subtotal_before_safety_bytes"), 708_949_459_392, "legacy subtotal")
    _same(value.get("legacy_required_safe_disk_bytes"), 1_063_424_189_088, "legacy total")
    _same(value.get("breakdown_closes_to_required_safe_bytes"), True, "legacy closure flag")
    rows = value.get("rows")
    if not isinstance(rows, list):
        _fail("original budget rows absent")
    if any(
        not isinstance(row, Mapping) or set(row) != set(ORIGINAL_BREAKDOWN_FIELDS)
        for row in rows
    ):
        _fail("original budget row schema changed")
    expected_components = {
        "raw lidar payload",
        "temporary download files",
        "decoded arrays",
        "map scan materialization",
        "accumulated target map",
        "voxel map copies",
        "query decoded points",
        "canonical source arrays",
        "canonical target arrays",
        "Open3D copies",
        "PCL copies",
        "per-snapshot target copies",
        "intermediate PCD/bin files",
        "geometry metrics",
        "publication artifacts",
        "resume/checkpoint files",
        "safety factor",
        "filesystem overhead",
    }
    _same({row.get("component") for row in rows}, expected_components, "original components")
    nonsafety = sum(
        int(row.get("peak_contribution_bytes", -1))
        for row in rows
        if row.get("component") != "safety factor"
    )
    total = sum(int(row.get("peak_contribution_bytes", -1)) for row in rows)
    _same(nonsafety, 708_949_459_392, "original nonsafety contribution")
    _same(total, 1_063_424_189_088, "original total contribution")
    _verify_csv_projection(
        root / "stage2_storage_budget_original_breakdown.csv",
        rows,
        ORIGINAL_BREAKDOWN_FIELDS,
    )


def _verify_lifecycle(root: Path) -> None:
    value = _load_json(root / "stage2_object_lifecycle_contract.json")
    _exact_keys(
        value,
        {"lifecycle_classes", "rows", "schema_version", "two_pass_query_preferred"},
        "lifecycle contract",
    )
    classes = [
        "REMOTE_ONLY",
        "TEMPORARY_DOWNLOAD",
        "TEMPORARY_DECODED",
        "PERSISTENT_RAW",
        "PERSISTENT_CANONICAL",
        "PERSISTENT_MAP",
        "PERSISTENT_RESULT",
        "CHECKPOINT",
    ]
    _same(value.get("lifecycle_classes"), classes, "lifecycle class declaration")
    _same(value.get("two_pass_query_preferred"), True, "two-pass lifecycle rule")
    rows = value.get("rows")
    if not isinstance(rows, list):
        _fail("lifecycle rows absent")
    if any(
        not isinstance(row, Mapping) or set(row) != set(LIFECYCLE_FIELDS)
        for row in rows
    ):
        _fail("lifecycle row schema changed")
    _same({row.get("lifecycle_class") for row in rows}, set(classes), "lifecycle coverage")
    for row in rows:
        life = row.get("lifecycle_class")
        persistent = row.get("persistent")
        if life in {"REMOTE_ONLY", "TEMPORARY_DOWNLOAD", "TEMPORARY_DECODED"}:
            _same(persistent, False, f"temporary lifecycle persistence {row.get('object_class')}")
        if life in {
            "PERSISTENT_RAW",
            "PERSISTENT_CANONICAL",
            "PERSISTENT_MAP",
            "PERSISTENT_RESULT",
            "CHECKPOINT",
        }:
            _same(persistent, True, f"persistent lifecycle persistence {row.get('object_class')}")
        if not str(row.get("resume_rule", "")).strip():
            _fail("empty lifecycle resume rule")
    target_rows = [row for row in rows if row.get("lifecycle_class") == "PERSISTENT_MAP"]
    _same(len(target_rows), 1, "persistent target lifecycle row count")
    if "one physical copy" not in str(target_rows[0].get("deletion_or_retention_rule")):
        _fail("persistent target lifecycle does not require one physical copy")
    _verify_csv_projection(
        root / "stage2_object_lifecycle_contract.csv", rows, LIFECYCLE_FIELDS
    )


def _verify_canonical_and_streaming(root: Path) -> None:
    canonical = _load_json(root / "canonical_bundle_storage_contract_v2.json")
    _exact_keys(
        canonical,
        {
            "actual_canonical_source_count",
            "actual_target_map_count",
            "backend_parameter_contract_sha256",
            "canonical_array_container",
            "canonicalizer_point_count_contract",
            "future_open3d_source_sha256",
            "future_open3d_target_sha256",
            "future_pcl_source_sha256",
            "future_pcl_target_sha256",
            "input_equality_requirements",
            "pcl_runtime_conversion",
            "prohibitions",
            "role",
            "schema_version",
            "shared_input_architecture",
        },
        "canonical bundle contract",
    )
    _same(canonical.get("actual_canonical_source_count"), 0, "actual canonical sources")
    _same(canonical.get("actual_target_map_count"), 0, "actual target maps")
    _same(
        canonical.get("canonicalizer_point_count_contract"),
        PREPROCESSING_UNRESOLVED,
        "canonicalizer point-count contract",
    )
    _same(
        canonical.get("backend_parameter_contract_sha256"),
        BACKEND_PARAMETER_CONTRACT_SHA256,
        "backend parameter binding",
    )
    for key, expected in (
        ("canonical_array_container", "NPY_V1_CONTENT_ADDRESSED_IMMUTABLE"),
        ("role", "FUTURE_STAGE2_STORAGE_CONTRACT_NOT_EXECUTION_AUTHORIZATION"),
        ("schema_version", "boreas_v2_canonical_bundle_storage_contract_v2"),
        (
            "shared_input_architecture",
            "ONE_CANONICAL_SOURCE_AND_TARGET_FAN_OUT_TO_BOTH_BACKENDS",
        ),
    ):
        _same(canonical.get(key), expected, f"canonical bundle {key}")
    source_ref = "REFERENCED_CONTENT_ADDRESSED_CANONICAL_SOURCE_SHA256"
    target_ref = "REFERENCED_CONTENT_ADDRESSED_TARGET_MAP_SHA256"
    _same(canonical.get("future_open3d_source_sha256"), source_ref, "Open3D source ref")
    _same(canonical.get("future_pcl_source_sha256"), source_ref, "PCL source ref")
    _same(canonical.get("future_open3d_target_sha256"), target_ref, "Open3D target ref")
    _same(canonical.get("future_pcl_target_sha256"), target_ref, "PCL target ref")
    equality = canonical.get("input_equality_requirements")
    _exact_keys(
        equality,
        {
            "future_open3d_source_sha256_equals_future_pcl_source_sha256",
            "future_open3d_target_sha256_equals_future_pcl_target_sha256",
        },
        "canonical input equality requirements",
    )
    if not all(value is True for value in equality.values()):
        _fail("canonical backend equality requirements changed")
    prohibitions = set(canonical.get("prohibitions", []))
    required_prohibitions = {
        "NO_PERSISTENT_OPEN3D_ONLY_SOURCE_COPY",
        "NO_PERSISTENT_PCL_ONLY_SOURCE_COPY",
        "NO_PERSISTENT_OPEN3D_ONLY_TARGET_COPY",
        "NO_PERSISTENT_PCL_ONLY_TARGET_COPY",
        "NO_PER_SNAPSHOT_TARGET_MAP_COPY",
    }
    _same(prohibitions, required_prohibitions, "canonical duplication prohibitions")
    pcl = canonical.get("pcl_runtime_conversion")
    _exact_keys(
        pcl,
        {"cleanup", "determinism", "directory", "persistent", "reverse_validation"},
        "PCL conversion contract",
    )
    _same(pcl.get("persistent"), False, "PCL conversion persistence")
    _same(pcl.get("cleanup"), "DELETE_IN_FINALLY_AFTER_RESULT_OR_ERROR", "PCL cleanup")
    _same(pcl.get("directory"), "tmp_pcl", "PCL temporary directory")
    _same(
        pcl.get("determinism"),
        "BYTE_IDENTICAL_FROM_SAME_CANONICAL_NPY_AND_CONVERSION_CONTRACT",
        "PCL conversion determinism",
    )
    _same(
        pcl.get("reverse_validation"),
        "TEMPORARY_PCD_REPARSE_MUST_EQUAL_THE_DETERMINISTIC_FLOAT32_PROJECTION_"
        "OF_THE_CANONICAL_ARRAY",
        "PCL reverse validation",
    )

    map_contract = _load_json(root / "stage2_streaming_map_contract.json")
    _exact_keys(
        map_contract,
        {
            "actual_map_object_processing_count",
            "actual_target_map_count",
            "algorithm",
            "allowlist_object_count",
            "allowlist_remote_bytes",
            "architecture_test_scope",
            "batch_incremental_semantics",
            "content_addressing",
            "crash_resume_semantics",
            "floating_point_risk",
            "flow",
            "local_subset_policy",
            "primary_map_sequence_id",
            "production_disk_checkpoint",
            "real_boreas_map_built",
            "role",
            "schema_version",
            "scientific_preprocessing_parameters",
            "storage_planner_parameter_authority",
        },
        "streaming map contract",
    )
    for key, expected in (
        ("actual_map_object_processing_count", 0),
        ("actual_target_map_count", 0),
        ("allowlist_object_count", EXPECTED_MAP_OBJECT_COUNT),
        ("allowlist_remote_bytes", EXPECTED_MAP_REMOTE_BYTES),
        ("primary_map_sequence_id", PRIMARY_MAP_SEQUENCE),
        ("real_boreas_map_built", False),
        ("storage_planner_parameter_authority", "NONE"),
        ("architecture_test_scope", "SYNTHETIC_TINY_POINT_CLOUDS_ONLY"),
        ("crash_resume_semantics", "AUTHENTICATED_STATE_CHAIN_PRODUCES_SAME_FINAL_BYTES"),
        ("role", "DRY_RUN_ARCHITECTURE_ONLY"),
        ("schema_version", "boreas_v2_stage2_streaming_map_contract_v1"),
    ):
        _same(map_contract.get(key), expected, f"streaming map {key}")
    content = map_contract.get("content_addressing")
    _exact_keys(
        content,
        {
            "planned_physical_target_map_copy_count",
            "planned_snapshot_target_reference_count",
            "planned_unique_target_map_count",
            "snapshot_target_storage",
        },
        "map content-addressing contract",
    )
    for key, expected in (
        ("planned_physical_target_map_copy_count", 1),
        ("planned_snapshot_target_reference_count", EXPECTED_SNAPSHOT_COUNT),
        ("planned_unique_target_map_count", 1),
        ("snapshot_target_storage", "SHA256_REFERENCE_ONLY"),
    ):
        _same(content.get(key), expected, f"map content addressing {key}")
    preprocessing = map_contract.get("scientific_preprocessing_parameters")
    if not isinstance(preprocessing, Mapping):
        _fail("map preprocessing contract absent")
    _same(
        set(preprocessing),
        {
            "association_distance",
            "normal_neighborhood",
            "range_limits",
            "voxel_origin_xyz_m",
            "voxel_size_m",
        },
        "map preprocessing fields",
    )
    for field, value in preprocessing.items():
        _same(value, PREPROCESSING_UNRESOLVED, f"unresolved preprocessing {field}")
    _exact_keys(
        map_contract.get("local_subset_policy"),
        {
            "default",
            "future_exception",
            "registration_result_adaptation",
            "subset_content_addressed_and_deduplicated",
        },
        "map local-subset policy",
    )
    algorithm = map_contract.get("algorithm")
    _exact_keys(
        algorithm,
        {
            "accumulator",
            "duplicate_handling",
            "final_point_order",
            "input_order",
            "numeric_dtype",
            "processing_order",
            "representative",
            "worker_count_effect",
        },
        "streaming map algorithm",
    )
    for key, expected in (
        ("accumulator", "VOXEL_KEY_TO_COUNT_AND_FLOAT64_SUM_XYZ"),
        ("duplicate_handling", "KEEP_ALL_FINITE_POINTS_IN_SUFFICIENT_STATISTICS"),
        ("final_point_order", "LEXICOGRAPHIC_VOXEL_KEY_X_Y_Z"),
        ("input_order", "FROZEN_ALLOWLIST_ORDER_WITH_EXPLICIT_ORDINAL"),
        ("numeric_dtype", "LITTLE_ENDIAN_FLOAT64"),
        ("processing_order", "SCAN_ORDINAL_THEN_POINT_ORDINAL"),
        ("representative", "CENTROID_FROM_FIXED_ORDER_FLOAT64_SUM"),
        ("worker_count_effect", "NONE_WORKERS_MAY_NOT_CHANGE_REDUCTION_ORDER"),
    ):
        _same(algorithm.get(key), expected, f"streaming map algorithm {key}")
    _same(
        map_contract.get("batch_incremental_semantics"),
        "BYTE_EXACT_FOR_SAME_ORDERED_INPUTS_AND_RULE",
        "batch/incremental semantics",
    )
    production_checkpoint = map_contract.get("production_disk_checkpoint")
    _exact_keys(
        production_checkpoint,
        {
            "budget_upper_bound_bytes",
            "format",
            "identity",
            "internal_state_sha_semantics",
            "ledger_and_receipt_overhead_budget_component",
            "reason",
            "replay_array_payload_bytes",
            "retention",
            "synthetic_json_checkpoint_scope",
            "target_npy_byte_exact_after_authenticated_replay",
        },
        "production map disk checkpoint contract",
    )
    for key, expected in (
        ("budget_upper_bound_bytes", EXPECTED_MAP_REMOTE_BYTES + 128),
        ("format", "ONE_PREALLOCATED_LITTLE_ENDIAN_FLOAT64_XYZ_REPLAY_ARRAY"),
        ("replay_array_payload_bytes", EXPECTED_MAP_REMOTE_BYTES),
        (
            "ledger_and_receipt_overhead_budget_component",
            "SMALL_EVIDENCE_UPPER_BOUND_NOT_THE_128_BYTE_ARRAY_ALLOWANCE",
        ),
        (
            "retention",
            "DELETE_ONLY_AFTER_FINAL_CONTENT_ADDRESSED_TARGET_COMMITS_AND_VERIFIES",
        ),
        (
            "synthetic_json_checkpoint_scope",
            "TINY_FIXTURE_ONLY_NOT_PRODUCTION_DISK_BUDGET",
        ),
        ("target_npy_byte_exact_after_authenticated_replay", True),
        (
            "internal_state_sha_semantics",
            "REPRESENTATION_SPECIFIC_NOT_COMPARED_BETWEEN_DIRECT_TRANSFORM_AND_"
            "PRETRANSFORMED_REPLAY",
        ),
    ):
        _same(
            production_checkpoint.get(key),
            expected,
            f"production map checkpoint {key}",
        )

    query = _load_json(root / "stage2_streaming_query_contract.json")
    _exact_keys(
        query,
        {
            "actual_geometry_metric_row_count",
            "actual_query_object_processing_count",
            "actual_selected_source_count",
            "allowlist_object_count",
            "allowlist_remote_bytes",
            "architecture_test_scope",
            "first_pass",
            "primary_query_sequence_id",
            "real_boreas_geometry_computed",
            "role",
            "schema_version",
            "second_pass",
            "selection_contract",
            "two_pass_adjudication",
        },
        "streaming query contract",
    )
    for key, expected in (
        ("actual_geometry_metric_row_count", 0),
        ("actual_query_object_processing_count", 0),
        ("actual_selected_source_count", 0),
        ("allowlist_object_count", EXPECTED_QUERY_OBJECT_COUNT),
        ("allowlist_remote_bytes", EXPECTED_QUERY_REMOTE_BYTES),
        ("primary_query_sequence_id", PRIMARY_QUERY_SEQUENCE),
        ("real_boreas_geometry_computed", False),
        ("architecture_test_scope", "SYNTHETIC_TINY_POINT_CLOUDS_ONLY"),
        ("role", "DRY_RUN_ARCHITECTURE_ONLY"),
        ("schema_version", "boreas_v2_stage2_streaming_query_contract_v1"),
    ):
        _same(query.get(key), expected, f"streaming query {key}")
    first = query.get("first_pass")
    second = query.get("second_pass")
    selection = query.get("selection_contract")
    two_pass = query.get("two_pass_adjudication")
    _exact_keys(
        first,
        {"blind_to_registration_results", "candidate_object_count", "persistent_fields", "raw_payload_cleanup"},
        "query first-pass contract",
    )
    _exact_keys(
        second,
        {
            "begins_only_after_R14_selection_freeze",
            "canonical_source_count",
            "checkpoint_file",
            "checkpoint_processing_binding",
            "checkpoint_record_kind",
            "object_identity_must_match_first_pass",
            "raw_payload_cleanup",
            "resume",
            "selected_object_count",
        },
        "query second-pass contract",
    )
    _exact_keys(
        selection,
        {
            "geometry_only_selector",
            "planned_rich_snapshot_count",
            "planned_snapshot_count",
            "planned_weak_snapshot_count",
            "registration_derived_fields_forbidden",
            "storage_planner_selects_snapshots",
        },
        "query selection contract",
    )
    _exact_keys(
        two_pass,
        {"R07", "R08", "R14", "preferred", "reproducibility", "sha_auditability"},
        "query two-pass adjudication",
    )
    _same(first.get("candidate_object_count"), EXPECTED_QUERY_OBJECT_COUNT, "pass-1 count")
    _same(first.get("blind_to_registration_results"), True, "pass-1 blindness")
    _same(
        first.get("raw_payload_cleanup"),
        "IMMEDIATE_AFTER_METRIC_AND_CHECKPOINT_COMMIT",
        "pass-1 cleanup",
    )
    _same(
        first.get("persistent_fields"),
        [
            "timestamp",
            "S3_key",
            "ETag",
            "LastModified",
            "remote_bytes",
            "object_identity_sha256",
            "geometry_only_metrics",
            "GT_sha256",
            "calibration_sha256",
            "target_map_sha256",
            "geometry_row_sha256",
        ],
        "pass-1 persistent fields",
    )
    _same(second.get("selected_object_count"), EXPECTED_SNAPSHOT_COUNT, "pass-2 count")
    _same(second.get("canonical_source_count"), EXPECTED_SNAPSHOT_COUNT, "pass-2 sources")
    _same(second.get("begins_only_after_R14_selection_freeze"), True, "R14 pass-2 gate")
    _same(
        second.get("checkpoint_file"),
        "processed_selected_sources.jsonl",
        "pass-2 checkpoint file",
    )
    _same(second.get("checkpoint_record_kind"), "SELECTED_SOURCE", "pass-2 record kind")
    _same(
        second.get("checkpoint_processing_binding"),
        "SHA256_OF_CANONICALIZATION_CONTRACT_AND_FROZEN_SELECTION_RECORD",
        "pass-2 checkpoint binding",
    )
    _same(
        second.get("resume"),
        "SKIP_ONLY_AUTHENTICATED_COMPLETED_SELECTION_PREFIX",
        "pass-2 resume",
    )
    _same(selection.get("planned_snapshot_count"), EXPECTED_SNAPSHOT_COUNT, "planned snapshots")
    _same(selection.get("planned_weak_snapshot_count"), 50, "planned weak snapshots")
    _same(selection.get("planned_rich_snapshot_count"), 50, "planned rich snapshots")
    _same(selection.get("registration_derived_fields_forbidden"), True, "geometry-only selection")
    _same(selection.get("storage_planner_selects_snapshots"), False, "storage selector authority")
    _same(two_pass.get("preferred"), True, "two-pass preference")
    for gate in ("R07", "R08", "R14", "reproducibility", "sha_auditability"):
        if not str(two_pass.get(gate, "")).startswith("PASS"):
            _fail(f"two-pass adjudication failed: {gate}")


def _verify_resume_and_temp(root: Path) -> None:
    checkpoint = _load_json(root / "stage2_checkpoint_contract.json")
    _exact_keys(
        checkpoint,
        {
            "actual_completed_map_object_count",
            "actual_completed_query_object_count",
            "actual_completed_selected_source_count",
            "append_semantics",
            "checkpoint_files",
            "completed_record_fields",
            "identity_change_semantics",
            "manifest_overwrite",
            "orphan_state",
            "partial_temp",
            "production_map_resume_payload",
            "resume",
            "schema_version",
            "strict_resume_implemented",
        },
        "checkpoint contract",
    )
    for key, expected in (
        ("actual_completed_map_object_count", 0),
        ("actual_completed_query_object_count", 0),
        ("actual_completed_selected_source_count", 0),
        ("append_semantics", "LOCKED_O_APPEND_CANONICAL_HASH_CHAIN_FILE_AND_DIRECTORY_FSYNC"),
        ("identity_change_semantics", "FAIL_ON_SIZE_ETAG_LASTMODIFIED_OR_CONTRACT_CHANGE"),
        ("manifest_overwrite", "FORBIDDEN"),
        ("orphan_state", "REJECT_NEVER_AUTO_ADOPT"),
        ("partial_temp", "DELETE_THEN_REACQUIRE_SAME_FROZEN_OBJECT_IDENTITY"),
        (
            "production_map_resume_payload",
            "ONE_PREALLOCATED_FLOAT64_XYZ_REPLAY_ARRAY_WITH_AUTHENTICATED_FIXED_ORDER_"
            "BYTE_RANGES",
        ),
        ("resume", "SKIP_ONLY_COMPLETE_AUTHENTICATED_CHAIN_MEMBERS"),
        ("strict_resume_implemented", True),
    ):
        _same(checkpoint.get(key), expected, f"checkpoint {key}")
    _same(
        checkpoint.get("checkpoint_files"),
        [
            "processed_map_objects.jsonl",
            "processed_query_objects.jsonl",
            "processed_selected_sources.jsonl",
        ],
        "checkpoint files",
    )
    required_fields = {
        "S3_key",
        "remote_size",
        "ETag",
        "LastModified",
        "local_temporary_SHA_if_materialized",
        "processing_contract_SHA",
        "GT_SHA",
        "calibration_SHA",
        "result_geometry_row_SHA_or_map_state_transition_SHA",
        "completed_at_UTC",
        "prior_record_SHA",
        "record_SHA",
    }
    _same(set(checkpoint.get("completed_record_fields", [])), required_fields, "checkpoint fields")
    _same(
        checkpoint.get("schema_version"),
        "boreas_v2_stage2_checkpoint_contract_v1",
        "checkpoint schema",
    )

    temp = _load_json(root / "stage2_temp_file_contract.json")
    _exact_keys(
        temp,
        {
            "directories",
            "frozen_manifest_inclusion",
            "persistent_directories",
            "rules",
            "safe_to_clear_when_validated",
            "schema_version",
        },
        "temporary-file contract",
    )
    _same(
        temp.get("directories"),
        {
            "tmp_decode": "TEMPORARY_DECODED",
            "tmp_download": "TEMPORARY_DOWNLOAD",
            "tmp_pcl": "TEMPORARY_DECODED_BACKEND_CONVERSION",
        },
        "temporary directory purposes",
    )
    _same(temp.get("frozen_manifest_inclusion"), False, "temp manifest exclusion")
    _same(temp.get("safe_to_clear_when_validated"), True, "temp safe-clear gate")
    _same(
        temp.get("schema_version"),
        "boreas_v2_stage2_temp_file_contract_v1",
        "temporary-file schema",
    )
    required_rules = {
        "EACH_TEMP_ROOT_HAS_EXPLICIT_MARKER_AND_PURPOSE",
        "NO_SYMLINK_OR_UNKNOWN_ENTRY_CLEANUP",
        "SUCCESSFUL_CONSUMPTION_DELETES_TEMP_IMMEDIATELY",
        "CRASH_RESUME_REMOVES_ONLY_MARKED_PARTIAL_TEMP_FILES",
        "TEMP_IS_NEVER_THE_ONLY_COPY_OF_SCIENTIFIC_EVIDENCE",
        "PERSISTENT_OUTPUT_MUST_COMMIT_BEFORE_TEMP_DELETE",
    }
    _same(set(temp.get("rules", [])), required_rules, "temporary cleanup rules")
    _same(
        set(temp.get("persistent_directories", [])),
        {
            "receipts",
            "target_map",
            "geometry_metrics",
            "selected_sources",
            "manifests",
            "results",
            "checkpoints",
        },
        "persistent directory contract",
    )


def _verify_science_and_architecture(root: Path) -> None:
    locks = _load_json(root / "stage2_scientific_lock_contract.json")
    _exact_keys(
        locks,
        {
            "allowlist_object_count",
            "future_success_design",
            "planner_configurable_fields",
            "preprocessing_parameter_status",
            "primary_pair",
            "remote_payload_bytes",
            "scientific_fields_planner_configurable",
            "scientific_locked_fields",
            "storage_only_changes",
            "voxel_size",
        },
        "scientific-lock contract",
    )
    _same(locks.get("scientific_fields_planner_configurable"), False, "science planner authority")
    _same(
        locks.get("planner_configurable_fields"),
        ["current_free_bytes", "execution_mode", "persistent_directory", "temporary_directory"],
        "planner configurable fields",
    )
    _same(
        locks.get("primary_pair"),
        {"map_sequence_id": PRIMARY_MAP_SEQUENCE, "query_sequence_id": PRIMARY_QUERY_SEQUENCE},
        "scientific lock primary pair",
    )
    _same(
        locks.get("allowlist_object_count"),
        EXPECTED_ALLOWLIST_OBJECT_COUNT,
        "lock object count",
    )
    _same(locks.get("remote_payload_bytes"), EXPECTED_REMOTE_BYTES, "lock remote bytes")
    _same(
        locks.get("preprocessing_parameter_status"),
        PREPROCESSING_UNRESOLVED,
        "lock preprocessing",
    )
    _same(locks.get("voxel_size"), PREPROCESSING_UNRESOLVED, "lock voxel size")
    design = locks.get("future_success_design")
    _same(
        design,
        {
            "weak_interval_count": 10,
            "rich_interval_count": 10,
            "snapshots_per_interval": 5,
            "weak_snapshot_count": 50,
            "rich_snapshot_count": 50,
            "snapshot_count": 100,
            "backend_trial_count": 200,
        },
        "future success design",
    )
    locked = set(locks.get("scientific_locked_fields", []))
    _same(
        locked,
        {
            "allowlist_object_count",
            "association_distance",
            "backend_trial_count",
            "geometry_metric",
            "icp_parameters",
            "normal_neighborhood",
            "primary_pair",
            "range_limits",
            "remote_payload_bytes",
            "reserve_pairs",
            "rich_interval_count",
            "rich_snapshot_count",
            "snapshot_count",
            "snapshots_per_interval",
            "voxel_size",
            "weak_interval_count",
            "weak_rich_thresholds",
            "weak_snapshot_count",
        },
        "scientific locked fields",
    )
    _same(
        locks.get("storage_only_changes"),
        [
            "data_lifecycle",
            "file_organization",
            "cache_policy",
            "content_addressing",
            "temporary_file_cleanup",
            "streaming_architecture",
        ],
        "storage-only change scope",
    )
    for field in (
        "allowlist_object_count",
        "remote_payload_bytes",
        "voxel_size",
        "range_limits",
        "normal_neighborhood",
        "association_distance",
        "geometry_metric",
        "weak_rich_thresholds",
        "snapshot_count",
        "icp_parameters",
    ):
        if field not in locked:
            _fail(f"scientific lock field absent: {field}")

    architecture = _load_json(root / "stage2_storage_architecture_contract.json")
    _exact_keys(
        architecture,
        {
            "PCL_conversion",
            "PCL_temporary_input_retained",
            "canonical_source_physical_copy_count_per_snapshot",
            "canonical_target_physical_copy_count",
            "execution_modes",
            "future_open3d_source_sha256_equals_future_pcl_source_sha256",
            "future_open3d_target_sha256_equals_future_pcl_target_sha256",
            "mode_scientific_equivalence_required",
            "per_snapshot_target_copy_allowed",
            "physical_target_map_copy_count",
            "preprocessing_parameter_status",
            "snapshot_target_binding",
            "snapshot_target_reference_count",
            "stage1_primary_pair",
            "target_store",
            "unique_target_map_count",
        },
        "storage architecture contract",
    )
    for key, expected in (
        ("unique_target_map_count", 1),
        ("snapshot_target_reference_count", EXPECTED_SNAPSHOT_COUNT),
        ("physical_target_map_copy_count", 1),
        ("per_snapshot_target_copy_allowed", False),
        ("canonical_source_physical_copy_count_per_snapshot", 1),
        ("canonical_target_physical_copy_count", 1),
        ("target_store", "target_maps/<target_map_sha256>/target_points.npy"),
        ("snapshot_target_binding", "target_map_sha256"),
        ("future_open3d_source_sha256_equals_future_pcl_source_sha256", True),
        ("future_open3d_target_sha256_equals_future_pcl_target_sha256", True),
        ("PCL_conversion", "DETERMINISTIC_TEMPORARY_DERIVATION_FROM_CANONICAL_NPY"),
        ("PCL_temporary_input_retained", False),
        ("preprocessing_parameter_status", PREPROCESSING_UNRESOLVED),
    ):
        _same(architecture.get(key), expected, f"storage architecture {key}")
    _same(
        architecture.get("mode_scientific_equivalence_required"),
        [
            "target_map_sha256",
            "geometry_metrics_sha256",
            "selection_manifest_sha256",
            "selected_canonical_source_sha256",
            "backend_results",
        ],
        "architecture mode-equivalence outputs",
    )
    _same(
        architecture.get("stage1_primary_pair"),
        {"map_sequence_id": PRIMARY_MAP_SEQUENCE, "query_sequence_id": PRIMARY_QUERY_SEQUENCE},
        "architecture primary pair",
    )
    execution_modes = architecture.get("execution_modes")
    if not isinstance(execution_modes, Mapping):
        _fail("architecture execution modes absent")
    _same(
        execution_modes.get("STREAMING_LOW_DISK"),
        {"persistent_raw_payload": False, "two_pass_query_selection": True},
        "streaming architecture mode",
    )
    _same(
        execution_modes.get("FULL_RAW_CACHE"),
        {"persistent_raw_payload": True, "two_pass_query_selection": True},
        "full-cache architecture mode",
    )


def _expected_estimates(stage1: Mapping[str, Any]) -> dict[str, int]:
    target = EXPECTED_MAP_REMOTE_BYTES
    selected = EXPECTED_SNAPSHOT_COUNT * (int(stage1["query_max_object_bytes"]) + 128)
    evidence = int(stage1["closure_bytes"]) + 19 * int(stage1["allowlist_bytes"])
    pcl_temp = (
        target // 2
        + 4096
        + int(stage1["query_max_object_bytes"]) // 2
        + 4096
        + 1024**2
    )
    one_scan = 4 * int(stage1["max_object_bytes"]) + 1024**2
    checkpoint = target + 128
    persistent = target + 128 + selected + evidence
    return {
        "target_map_upper_bound_bytes": target + 128,
        "selected_sources_upper_bound_bytes": selected,
        "small_evidence_upper_bound_bytes": evidence,
        "pcl_runtime_conversion_upper_bound_bytes": pcl_temp,
        "single_scan_streaming_upper_bound_bytes": one_scan,
        "map_checkpoint_upper_bound_bytes": checkpoint,
        "persistent_without_raw_bytes": persistent,
    }


def _expected_mode(
    mode: str,
    persistent: int,
    temporary: int,
    safety: float,
    current: int,
) -> dict[str, Any]:
    live = persistent + temporary
    margin = math.ceil(live * safety)
    recommended = live + margin
    return {
        "mode": mode,
        "persistent_bytes": persistent,
        "temporary_peak_bytes": temporary,
        "simultaneously_live_peak_bytes": live,
        "safety_margin_fraction": safety,
        "safety_margin_bytes": margin,
        "recommended_free_disk_bytes": recommended,
        "persistent_GiB": persistent / GIB,
        "temporary_peak_GiB": temporary / GIB,
        "simultaneously_live_peak_GiB": live / GIB,
        "safety_margin_GiB": margin / GIB,
        "recommended_free_disk_GiB": recommended / GIB,
        "current_free_bytes": current,
        "CURRENT_DISK_SUFFICIENT": current >= recommended,
        "additional_bytes_required": max(0, recommended - current),
        "additional_GiB_required": max(0, recommended - current) / GIB,
        "minimum_free_disk_required_before_start_bytes": recommended,
        "abort_if_free_disk_below_bytes": margin,
        "runtime_low_disk_watermark_bytes": margin,
        "watermark_definition": (
            "remaining free bytes after the projected next atomic write must be at least "
            "the mode safety margin"
        ),
        "watermark_action": "FAIL_CLOSED_BEFORE_NEXT_WRITE",
    }


def _verify_budget(root: Path, stage1: Mapping[str, Any]) -> Mapping[str, Any]:
    budget = _load_json(root / "boreas_v2_stage2_disk_budget_optimized.json")
    _exact_keys(
        budget,
        {
            "CURRENT_DISK_SUFFICIENT",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
            "PUBLIC_DATA_V2_RUN_AUTHORIZED",
            "REAL_REGISTRATION_AUTHORIZED",
            "STAGE2_DOWNLOAD_AUTHORIZED",
            "abort_if_free_disk_below_bytes",
            "accounting_rule",
            "downloaded_lidar_bytes",
            "downloaded_lidar_payload_count",
            "estimate_limitations",
            "minimum_additional_GiB_required",
            "minimum_free_disk_required_before_start_bytes",
            "modes",
            "recommended_additional_GiB_required",
            "registration_execution_count",
            "runtime_low_disk_watermark_bytes",
            "schema_version",
            "size_estimates",
        },
        "optimized budget contract",
    )
    _same(
        budget.get("schema_version"),
        "boreas_v2_stage2_storage_budget_optimized_v1",
        "optimized budget schema",
    )
    estimates = _expected_estimates(stage1)
    _same(budget.get("size_estimates"), estimates, "independent size estimates")
    modes = budget.get("modes")
    if not isinstance(modes, list) or len(modes) != 3:
        _fail("optimized budget must contain exactly three modes")
    current_values = {row.get("current_free_bytes") for row in modes}
    if len(current_values) != 1:
        _fail("optimized budget modes disagree on current free bytes")
    current = next(iter(current_values))
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        _fail("invalid recorded current free bytes")
    persistent = estimates["persistent_without_raw_bytes"]
    theoretical_temp = estimates["pcl_runtime_conversion_upper_bound_bytes"]
    operational_temp = max(
        estimates["map_checkpoint_upper_bound_bytes"],
        estimates["pcl_runtime_conversion_upper_bound_bytes"],
        estimates["single_scan_streaming_upper_bound_bytes"],
    )
    expected = [
        _expected_mode("THEORETICAL_MINIMUM", persistent, theoretical_temp, 0.10, current),
        _expected_mode("RECOMMENDED_OPERATIONAL", persistent, operational_temp, 0.20, current),
        _expected_mode(
            "CONSERVATIVE_FULL_CACHE",
            persistent + EXPECTED_REMOTE_BYTES,
            operational_temp,
            0.20,
            current,
        ),
    ]
    _same(modes, expected, "optimized budget arithmetic")
    by_mode = {row["mode"]: row for row in expected}
    recommended = by_mode["RECOMMENDED_OPERATIONAL"]
    minimum = by_mode["THEORETICAL_MINIMUM"]
    for key, expected_value in (
        ("CURRENT_DISK_SUFFICIENT", recommended["CURRENT_DISK_SUFFICIENT"]),
        ("minimum_additional_GiB_required", minimum["additional_GiB_required"]),
        ("recommended_additional_GiB_required", recommended["additional_GiB_required"]),
        (
            "minimum_free_disk_required_before_start_bytes",
            recommended["recommended_free_disk_bytes"],
        ),
        ("abort_if_free_disk_below_bytes", recommended["abort_if_free_disk_below_bytes"]),
        ("runtime_low_disk_watermark_bytes", recommended["runtime_low_disk_watermark_bytes"]),
    ):
        _same(budget.get(key), expected_value, f"optimized budget top-level {key}")
    for field in AUTHORIZATION_FIELDS:
        _same(budget.get(field), False, f"optimized budget {field}")
    for field in (
        "downloaded_lidar_payload_count",
        "downloaded_lidar_bytes",
        "registration_execution_count",
    ):
        _same(budget.get(field), 0, f"optimized budget {field}")
    limitations = budget.get("estimate_limitations")
    _exact_keys(
        limitations,
        {
            "actual_target_map_size",
            "map_checkpoint_budget_method",
            "metadata_budget_method",
            "preprocessing_parameter_status",
            "selected_source_budget_method",
            "target_map_budget_method",
        },
        "optimized budget limitations",
    )
    _same(
        limitations.get("preprocessing_parameter_status"),
        PREPROCESSING_UNRESOLVED,
        "budget preprocessing status",
    )
    target_method = str(limitations.get("target_map_budget_method", ""))
    source_method = str(limitations.get("selected_source_budget_method", ""))
    if "no voxel size is assumed" not in target_method:
        _fail("budget silently assumes a voxel size")
    if "preserve or reduce point count" not in target_method:
        _fail("target upper bound omits its point-count assumption")
    if "preserve or reduce point count" not in source_method:
        _fail("selected-source upper bound omits its point-count assumption")
    _verify_csv_projection(
        root / "boreas_v2_stage2_disk_budget_optimized.csv",
        modes,
        BUDGET_CSV_FIELDS,
    )
    return budget


def _verify_modes(root: Path, budget: Mapping[str, Any]) -> None:
    value = _load_json(root / "boreas_v2_stage2_execution_modes.json")
    _exact_keys(
        value,
        {
            "actual_execution_mode",
            "architecture_only",
            "mode_count",
            "mode_scientific_equivalence_required",
            "modes",
            "optimized_budget_contract_sha256",
            "real_execution_performed",
            "schema_version",
            "synthetic_fixture_equivalence",
        },
        "execution-modes contract",
    )
    for key, expected in (
        ("actual_execution_mode", None),
        ("architecture_only", True),
        ("mode_count", 2),
        ("mode_scientific_equivalence_required", True),
        ("real_execution_performed", False),
        ("synthetic_fixture_equivalence", "PASS"),
    ):
        _same(value.get(key), expected, f"execution modes {key}")
    _same(
        value.get("optimized_budget_contract_sha256"),
        _compact_sha256(budget),
        "execution modes budget binding",
    )
    modes = value.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != {"STREAMING_LOW_DISK", "FULL_RAW_CACHE"}:
        _fail("execution mode names changed")
    low = modes["STREAMING_LOW_DISK"]
    full = modes["FULL_RAW_CACHE"]
    mode_keys = {
        "budget_mode",
        "canonical_source_semantics",
        "geometry_metric_semantics",
        "raw_policy",
        "science_contract_fingerprint",
        "selection_semantics",
        "target_map_semantics",
        "target_policy",
    }
    _exact_keys(low, mode_keys, "streaming execution mode")
    _exact_keys(full, mode_keys, "full-cache execution mode")
    _same(low.get("budget_mode"), "RECOMMENDED_OPERATIONAL", "streaming budget mode")
    _same(full.get("budget_mode"), "CONSERVATIVE_FULL_CACHE", "full-cache budget mode")
    for field in (
        "canonical_source_semantics",
        "geometry_metric_semantics",
        "science_contract_fingerprint",
        "selection_semantics",
        "target_map_semantics",
    ):
        _same(low.get(field), full.get(field), f"mode semantic equivalence {field}")
    _same(low.get("target_policy"), "ONE_CONTENT_ADDRESSED_TARGET", "streaming target policy")
    _same(full.get("target_policy"), "ONE_CONTENT_ADDRESSED_TARGET", "full target policy")


def _verify_attestations(root: Path) -> None:
    no_lidar = _load_json(root / "NO_LIDAR_PAYLOAD_ATTESTATION.json")
    _exact_keys(
        no_lidar,
        set(AUTHORIZATION_FIELDS)
        | set(ZERO_EXECUTION_FIELDS)
        | {
            "NO_LIDAR_PAYLOAD_DOWNLOAD",
            "allowed_metadata_operation_count",
            "blocked_http_attempt_count",
            "blocked_lidar_payload_attempt_count",
            "blocked_process_attempt_count",
            "data_root",
            "guard_activation_count",
            "guard_was_activated",
            "host_wide_network_monitor",
            "initial_local_lidar_bin_count",
            "lidar_payload_download_bytes",
            "lidar_payload_download_count",
            "local_lidar_bin_count",
            "metadata_only_audit",
            "monitoring_scope",
            "no_lidar_payload_environment_active",
            "no_registration_environment_active",
            "pass",
            "payload_materialization_authorized",
            "status",
        },
        "NO-LiDAR attestation",
    )
    _require_zero_contract(no_lidar, "NO-LiDAR attestation")
    for field in (
        "lidar_payload_download_count",
        "lidar_payload_download_bytes",
        "downloaded_lidar_object_count",
        "downloaded_lidar_payload_count",
        "downloaded_lidar_bytes",
        "local_lidar_bin_count",
        "initial_local_lidar_bin_count",
        "blocked_lidar_payload_attempt_count",
        "blocked_process_attempt_count",
        "blocked_http_attempt_count",
    ):
        _same(no_lidar.get(field), 0, f"NO-LiDAR {field}")
    for field in (
        "NO_LIDAR_PAYLOAD_DOWNLOAD",
        "no_registration_environment_active",
        "no_lidar_payload_environment_active",
        "guard_was_activated",
        "metadata_only_audit",
        "pass",
    ):
        _same(no_lidar.get(field), True, f"NO-LiDAR {field}")
    _same(no_lidar.get("payload_materialization_authorized"), False, "payload authorization")
    _same(no_lidar.get("status"), "PASS", "NO-LiDAR status")

    no_icp = _load_json(root / "NO_ICP_ATTESTATION.json")
    _exact_keys(
        no_icp,
        set(AUTHORIZATION_FIELDS)
        | set(ZERO_EXECUTION_FIELDS)
        | {
            "estimated_transform_count",
            "estimated_transform_evidence",
            "estimated_transform_file_count",
            "estimated_transform_files",
            "open3d_registration_call_count",
            "other_registration_process_count",
            "pass",
            "pcl_cli_invocation_count",
            "static_source_audit",
            "status",
            "structured_result_scan_error_count",
            "structured_result_scan_error_files",
        },
        "NO-ICP attestation",
    )
    _require_zero_contract(no_icp, "NO-ICP attestation")
    for field in (
        "estimated_transform_count",
        "estimated_transform_file_count",
        "open3d_registration_call_count",
        "other_registration_process_count",
        "pcl_cli_invocation_count",
        "real_trial_result_count",
        "registration_execution_count",
        "structured_result_scan_error_count",
    ):
        _same(no_icp.get(field), 0, f"NO-ICP {field}")
    _same(no_icp.get("pass"), True, "NO-ICP pass")
    _same(no_icp.get("status"), "PASS", "NO-ICP status")
    static_audit = _exact_keys(
        no_icp.get("static_source_audit"),
        {"pass", "python_file_count", "violations"},
        "NO-ICP static source audit",
    )
    _same(static_audit.get("pass"), True, "NO-ICP static source pass")
    _same(static_audit.get("violations"), [], "NO-ICP static source violations")
    python_file_count = static_audit.get("python_file_count")
    if (
        isinstance(python_file_count, bool)
        or not isinstance(python_file_count, int)
        or python_file_count < 1
    ):
        _fail("NO-ICP static source Python file count is invalid")
    for field in (
        "estimated_transform_evidence",
        "estimated_transform_files",
        "structured_result_scan_error_files",
    ):
        _same(no_icp.get(field), [], f"NO-ICP empty evidence list {field}")


def _verify_readiness_and_summary(
    root: Path,
    budget: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    status = _verify_pytest_evidence(root, manifest)
    readiness = _load_json(root / "boreas_v2_stage2_storage_readiness.json")
    readiness_keys = set(AUTHORIZATION_FIELDS) | set(ZERO_EXECUTION_FIELDS) | {
        "CURRENT_DISK_SUFFICIENT",
        "STAGE2_STORAGE_PLAN_READY",
        "architecture_test_scope",
        "audit_pre_change_source_only_baseline",
        "canonical_backend_input_no_duplication",
        "checkpoint_resume_contract_ready",
        "current_free_bytes",
        "independent_storage_verifier_required",
        "next_step_requires_separate_user_authorization",
        "planned_future_backend_trial_count_after_success",
        "planned_future_rich_interval_count",
        "planned_future_rich_snapshot_count",
        "planned_future_snapshot_count",
        "planned_future_snapshots_per_interval",
        "planned_future_weak_interval_count",
        "planned_future_weak_snapshot_count",
        "preprocessing_parameter_status",
        "real_stage2_execution_runner_in_scope",
        "recommended_execution_mode",
        "repository_gate",
        "runtime_disk_gate_contract_ready",
        "runtime_disk_gate_runner_integration_required",
        "runtime_disk_gate_status",
        "schema_version",
        "source_only_test_status",
        "stage1_binding",
        "stage1_frozen_assets_verified",
        "storage_planner_verifier_command",
        "target_physical_copy_count",
    }
    _exact_keys(readiness, readiness_keys, "storage readiness")
    _require_zero_contract(readiness, "readiness")
    expected_ready = bool(budget["CURRENT_DISK_SUFFICIENT"])
    for key, expected in (
        ("CURRENT_DISK_SUFFICIENT", expected_ready),
        ("STAGE2_STORAGE_PLAN_READY", expected_ready),
        ("canonical_backend_input_no_duplication", True),
        ("checkpoint_resume_contract_ready", True),
        ("independent_storage_verifier_required", True),
        ("next_step_requires_separate_user_authorization", True),
        ("planned_future_rich_snapshot_count", 50),
        ("planned_future_snapshot_count", 100),
        ("planned_future_backend_trial_count_after_success", 200),
        ("planned_future_rich_interval_count", 10),
        ("planned_future_snapshots_per_interval", 5),
        ("planned_future_weak_interval_count", 10),
        ("planned_future_weak_snapshot_count", 50),
        ("preprocessing_parameter_status", PREPROCESSING_UNRESOLVED),
        ("recommended_execution_mode", "STREAMING_LOW_DISK"),
        ("real_stage2_execution_runner_in_scope", False),
        ("runtime_disk_gate_contract_ready", True),
        ("runtime_disk_gate_runner_integration_required", True),
        (
            "runtime_disk_gate_status",
            "PLANNER_START_AND_PROJECTED_WRITE_GUARDS_IMPLEMENTED_AND_SYNTHETICALLY_"
            "TESTED;SEPARATELY_AUTHORIZED_REAL_STAGE2_RUNNER_MUST_INVOKE_THEM",
        ),
        ("stage1_frozen_assets_verified", True),
        ("target_physical_copy_count", 1),
    ):
        _same(readiness.get(key), expected, f"readiness {key}")
    repository_gate = readiness.get("repository_gate")
    _exact_keys(
        repository_gate,
        {"branch", "head", "protected_tags", "worktree_clean"},
        "readiness repository gate",
    )
    _same(repository_gate.get("branch"), EXPECTED_BRANCH, "recorded branch")
    _same(repository_gate.get("head"), manifest.get("producer_commit"), "recorded HEAD")
    _same(repository_gate.get("protected_tags"), PROTECTED_TAGS, "recorded protected tags")
    _same(repository_gate.get("worktree_clean"), True, "recorded clean worktree")
    modes = {row["mode"]: row for row in budget["modes"]}
    _same(
        readiness.get("current_free_bytes"),
        modes["RECOMMENDED_OPERATIONAL"]["current_free_bytes"],
        "readiness current disk binding",
    )
    _same(readiness.get("source_only_test_status"), status, "readiness JUnit binding")
    _same(
        readiness.get("audit_pre_change_source_only_baseline"),
        {
            "collected": 811,
            "errors": 0,
            "failed": 0,
            "head": STAGE1_TAG_COMMIT,
            "passed": 801,
            "skipped": 10,
            "status": "PASS",
        },
        "pre-change source-only baseline",
    )

    binding = readiness.get("stage1_binding")
    if not isinstance(binding, Mapping):
        _fail("readiness Stage-1 binding absent")
    expected_binding = {
        "allowlist_object_count": EXPECTED_ALLOWLIST_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "allowlist_sha256": STAGE1_ALLOWLIST_SHA256,
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
        "frozen_manifest_file_sha256": STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "manifest_root_sha256": STAGE1_MANIFEST_ROOT_SHA256,
        "map_object_count": EXPECTED_MAP_OBJECT_COUNT,
        "map_remote_bytes": EXPECTED_MAP_REMOTE_BYTES,
        "max_object_bytes": 5_591_160,
        "no_icp_attestation_sha256": STAGE1_NO_ICP_SHA256,
        "no_lidar_attestation_sha256": STAGE1_NO_LIDAR_SHA256,
        "pair_selection_sha256": STAGE1_PAIR_SELECTION_SHA256,
        "primary_pair": {
            "map_sequence_id": PRIMARY_MAP_SEQUENCE,
            "query_sequence_id": PRIMARY_QUERY_SEQUENCE,
        },
        "query_object_count": EXPECTED_QUERY_OBJECT_COUNT,
        "query_remote_bytes": EXPECTED_QUERY_REMOTE_BYTES,
        "signed_file_count": 27,
        "top_100_query_remote_bytes": 556_548_720,
        "verification_pass": True,
    }
    _same(binding, expected_binding, "readiness Stage-1 binding")

    summary = _load_json(root / "stage2_storage_summary.json")
    _exact_keys(
        summary,
        readiness_keys | {"answer_count", "answers", "final_conclusion"},
        "storage summary",
    )
    _require_zero_contract(summary, "summary")
    for key in (
        "CURRENT_DISK_SUFFICIENT",
        "STAGE2_STORAGE_PLAN_READY",
        "planned_future_rich_snapshot_count",
        "planned_future_snapshot_count",
        "planned_future_weak_snapshot_count",
        "preprocessing_parameter_status",
        "recommended_execution_mode",
        "target_physical_copy_count",
    ):
        _same(summary.get(key), readiness.get(key), f"summary/readiness {key}")
    answers = summary.get("answers")
    if not isinstance(answers, list):
        _fail("summary answers absent")
    if any(
        not isinstance(row, Mapping) or set(row) != {"answer", "question_number"}
        for row in answers
    ):
        _fail("summary answer schema changed")
    _same(summary.get("answer_count"), 27, "summary answer count")
    _same(
        [row.get("question_number") for row in answers],
        list(range(1, 28)),
        "summary answer numbering",
    )
    answer_text = {
        int(row["question_number"]): str(row.get("answer", ""))
        for row in answers
    }
    required_answer_tokens = {
        18: ("locked O_APPEND canonical JSONL hash chain",),
        22: ("lidar/*.bin", "对象数和字节数均为 0"),
        23: ("Open3D/PCL/其他 registration", "均为 0"),
        24: (
            f"{status['collected']} collected",
            f"{status['passed']} passed",
            f"{status['skipped']} skipped",
            "0 failed, 0 errors",
        ),
        25: (
            "python3 scripts/verify_boreas_v2_stage2_storage_plan.py",
            "PASS",
            "11 个语义篡改",
        ),
        26: (f"STAGE2_STORAGE_PLAN_READY={str(expected_ready).lower()}",),
        27: ("单独任务", "STAGE2_DOWNLOAD_AUTHORIZED=false"),
    }
    for question_number, tokens in required_answer_tokens.items():
        for token in tokens:
            if token not in answer_text[question_number]:
                _fail(
                    f"summary answer {question_number} omits required token: {token}"
                )
    markdown = (root / "stage2_storage_summary.md").read_text(encoding="utf-8")
    ready_text = str(expected_ready).lower()
    for token in (
        f"STAGE2_STORAGE_PLAN_READY={ready_text}",
        "STAGE2_DOWNLOAD_AUTHORIZED=false",
        "PUBLIC_DATA_V2_RUN_AUTHORIZED=false",
        "REAL_REGISTRATION_AUTHORIZED=false",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false",
        "真实 Stage-2 runner 不在本任务范围",
        "每次 projected write 前接入已测试的磁盘门禁",
    ):
        if token not in markdown and token not in str(summary.get("final_conclusion", "")):
            _fail(f"summary boundary token absent: {token}")
    return readiness


def _verify_pytest_evidence(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently count every testcase in the frozen raw JUnit bytes."""

    xml_path = root / "source_only_pytest_junit.xml"
    try:
        xml_root = ET.fromstring(xml_path.read_bytes())
    except ET.ParseError as error:
        raise BoreasStage2StorageVerificationError("invalid frozen pytest JUnit XML") from error
    suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
    if not suites:
        _fail("frozen pytest JUnit contains no test suite")
    try:
        collected = sum(int(suite.attrib["tests"]) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    except (KeyError, ValueError) as error:
        raise BoreasStage2StorageVerificationError("invalid frozen pytest JUnit counts") from error
    testcases = [case for suite in suites for case in suite.findall("testcase")]
    testcase_count = len(testcases)
    identities = [
        (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        for case in testcases
    ]
    child_failures = sum(case.find("failure") is not None for case in testcases)
    child_errors = sum(case.find("error") is not None for case in testcases)
    child_skipped = sum(case.find("skipped") is not None for case in testcases)
    passed = testcase_count - child_failures - child_errors - child_skipped
    if (
        collected < 811
        or testcase_count != collected
        or any(not classname or not name for classname, name in identities)
        or len(set(identities)) != testcase_count
        or child_failures != failures
        or child_errors != errors
        or child_skipped != skipped
        or failures != 0
        or errors != 0
        or passed < 0
        or passed + skipped != collected
    ):
        _fail("frozen source-only pytest JUnit is not a clean >=811-test baseline")
    producer_head = manifest.get("producer_commit")
    expected = {
        "collected": collected,
        "errors": errors,
        "failed": failures,
        "head": producer_head,
        "junit_filename": "source_only_pytest_junit.xml",
        "junit_verification_scope": (
            "INDEPENDENTLY_PARSED_SHA_BOUND_COUNTS_AND_UNIQUE_TESTCASE_IDENTITIES;"
            "THE_XML_IS_NOT_A_SIGNED_ATTESTATION"
        ),
        "junit_sha256": _sha256_file(xml_path),
        "junit_size_bytes": xml_path.stat().st_size,
        "minimum_collected_required": 811,
        "passed": passed,
        "skipped": skipped,
        "status": "PASS",
        "testcase_count": testcase_count,
    }
    recorded = _load_json(root / "source_only_test_status.json")
    _same(recorded, expected, "independently recomputed pytest status")
    return expected


def _verify_no_local_lidar(data_root: Path | None) -> int:
    if data_root is None:
        return 0
    root = data_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        _fail("Boreas data root is absent or unsafe")
    found = [
        path
        for path in root.rglob("*.bin")
        if path.is_file() and path.parent.name == "lidar"
    ]
    if found:
        _fail(f"real Boreas lidar payload present locally: {len(found)} file(s)")
    return 0


def verify_boreas_stage2_storage_optimization(
    *,
    repository: str | Path,
    runtime_root: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact bytes, independent arithmetic, and storage-only semantics."""

    repository_path = Path(repository).resolve(strict=True)
    root = Path(runtime_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        _fail("Stage-2 storage root is absent or unsafe")
    data_path = Path(data_root) if data_root is not None else None

    stage1 = _verify_stage1(repository_path)
    manifest = _verify_manifest(root)
    _verify_original_budget(root)
    _verify_lifecycle(root)
    _verify_canonical_and_streaming(root)
    _verify_resume_and_temp(root)
    _verify_science_and_architecture(root)
    budget = _verify_budget(root, stage1)
    _verify_modes(root, budget)
    _verify_attestations(root)
    readiness = _verify_readiness_and_summary(root, budget, manifest)
    local_lidar_count = _verify_no_local_lidar(data_path)

    # This only proves that the recorded producer commit is a real repository
    # object and that frozen Stage-1 precedes it; it does not trust producer code.
    producer = str(manifest["producer_commit"])
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{producer}^{{commit}}"],
        cwd=repository_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        _fail("manifest producer commit does not exist")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STAGE1_TAG_COMMIT, producer],
        cwd=repository_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        _fail("frozen Stage-1 tag is not an ancestor of the producer commit")
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    producer_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", producer, current_head],
        cwd=repository_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if producer_ancestor.returncode != 0:
        _fail("storage producer commit is not in current HEAD history")
    for tag, expected_commit in PROTECTED_TAGS.items():
        actual_commit = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"],
            cwd=repository_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        _same(actual_commit, expected_commit, f"live protected tag {tag}")

    modes = {row["mode"]: row for row in budget["modes"]}
    return {
        "BOREAS_V2_STAGE2_STORAGE_VERIFICATION_PASS": True,
        "STAGE2_STORAGE_PLAN_VERIFICATION_PASS": True,
        "CURRENT_DISK_SUFFICIENT": budget["CURRENT_DISK_SUFFICIENT"],
        "STAGE2_STORAGE_PLAN_READY": readiness["STAGE2_STORAGE_PLAN_READY"],
        "allowlist_object_count": EXPECTED_ALLOWLIST_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "closure_file_count": len(REQUIRED_FILES),
        "local_lidar_bin_count": local_lidar_count,
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "optimized_modes": {
            name: {
                "CURRENT_DISK_SUFFICIENT": row["CURRENT_DISK_SUFFICIENT"],
                "recommended_free_disk_GiB": row["recommended_free_disk_GiB"],
            }
            for name, row in modes.items()
        },
        "physical_target_map_copy_count": 1,
        "planned_snapshot_count": EXPECTED_SNAPSHOT_COUNT,
        "real_execution_count": 0,
        "registration_execution_count": 0,
        "lidar_payload_download_count": 0,
        "verification_pass": True,
    }


def verify_boreas_v2_stage2_storage_plan(
    *,
    repository: str | Path,
    runtime_root: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Public task-named alias for the independent storage verifier."""

    return verify_boreas_stage2_storage_optimization(
        repository=repository,
        runtime_root=runtime_root,
        data_root=data_root,
    )


__all__ = [
    "BoreasStage2StorageVerificationError",
    "PAYLOAD_FILES",
    "REQUIRED_FILES",
    "verify_boreas_stage2_storage_optimization",
    "verify_boreas_v2_stage2_storage_plan",
]
