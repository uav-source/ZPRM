"""Build the metadata-only Boreas v2 Stage-2 storage-plan closure.

This module is an execution-architecture audit.  It consumes only the frozen
Stage-1 metadata closure and filesystem capacity metadata.  It must never
materialize a Boreas ``lidar/*.bin`` object, run a registration backend, build
a real target map, compute geometry metrics, or select snapshots.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    compact_sha256,
    sha256_file,
)
from .stage2_payload_guard import BoreasLidarPayloadGuard
from .stage2_storage_planner import (
    GIB,
    LIFECYCLE_FIELDS as PLANNER_LIFECYCLE_FIELDS,
    ORIGINAL_BREAKDOWN_FIELDS as PLANNER_ORIGINAL_BREAKDOWN_FIELDS,
    load_frozen_stage1_inputs,
    object_lifecycle_contract,
    optimized_storage_budgets,
    original_budget_breakdown,
    scientific_lock_contract,
    storage_architecture_contract,
)


EXPECTED_BRANCH = "prep/boreas-v2-stage2-storage-optimization"
STAGE1_RELATIVE_ROOT = Path(
    "frozen_assets/public_data_external_validation_v2_boreas_stage1"
)
STAGE1_TAG = "preparation/public-data-external-v2-boreas-stage1"
EXPECTED_TAG_COMMITS = {
    "archive/public-data-v1-screening-closed": (
        "59a1fa000d1de7fc4df2b7feeb5b4fb2f617750c"
    ),
    STAGE1_TAG: "fb6d84fc41a252fc1934716f96473fad1634ebdc",
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
PLANNED_SNAPSHOT_COUNT = 100
PLANNED_WEAK_SNAPSHOT_COUNT = 50
PLANNED_RICH_SNAPSHOT_COUNT = 50
PREPROCESSING_UNRESOLVED = "PREPROCESSING_PARAMETER_REQUIRES_STAGE2_PREREGISTRATION"

REQUIRED_ENVIRONMENT = (
    "ZPRM_REAL_DATA_PREP_NO_REGISTRATION",
    "ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD",
)

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
    "stage",
    "lifecycle",
    "location_class",
    "persistent",
    "identity_fields",
    "consumer",
    "cleanup_rule",
    "resume_rule",
    "scientific_evidence_role",
)


class BoreasStage2StorageOptimizationError(RuntimeError):
    """A storage-plan safety, lineage, or immutability gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _verify_sha256sums(root: Path) -> list[dict[str, Any]]:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise BoreasStage2StorageOptimizationError(f"unsafe or absent SHA256SUMS: {root}")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            raise BoreasStage2StorageOptimizationError("malformed Stage-1 SHA256SUMS")
        expected, name = match.groups()
        if name in names:
            raise BoreasStage2StorageOptimizationError("duplicate Stage-1 SHA256SUMS row")
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise BoreasStage2StorageOptimizationError(f"Stage-1 SHA mismatch: {name}")
        names.add(name)
        rows.append({"path": name, "sha256": expected, "size_bytes": path.stat().st_size})
    return rows


def _require_exact_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise BoreasStage2StorageOptimizationError(
            f"frozen Stage-1 hash changed: {path.name}: {actual} != {expected}"
        )


def inspect_frozen_stage1(repository: str | Path) -> dict[str, Any]:
    """Independently bind the storage audit to the frozen Stage-1 closure."""

    repository_path = Path(repository).resolve(strict=True)
    root = repository_path / STAGE1_RELATIVE_ROOT
    if root.is_symlink() or not root.is_dir():
        raise BoreasStage2StorageOptimizationError("frozen Stage-1 root is absent or unsafe")
    signed_rows = _verify_sha256sums(root)
    expected_hashes = {
        "frozen_manifest.json": STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "boreas_v2_pair_selection.json": STAGE1_PAIR_SELECTION_SHA256,
        "boreas_v2_stage2_download_allowlist.csv": STAGE1_ALLOWLIST_SHA256,
        "boreas_v2_stage2_download_plan.json": STAGE1_DOWNLOAD_PLAN_SHA256,
        "boreas_v2_stage2_disk_budget.json": STAGE1_OLD_BUDGET_SHA256,
        "NO_ICP_ATTESTATION.json": STAGE1_NO_ICP_SHA256,
        "NO_LIDAR_PAYLOAD_ATTESTATION.json": STAGE1_NO_LIDAR_SHA256,
    }
    for name, expected in expected_hashes.items():
        _require_exact_sha(root / name, expected)

    manifest = _load_json(root / "frozen_manifest.json")
    if manifest.get("manifest_root_sha256") != STAGE1_MANIFEST_ROOT_SHA256:
        raise BoreasStage2StorageOptimizationError("Stage-1 manifest root changed")
    pair = _load_json(root / "boreas_v2_pair_selection.json")
    primary = pair.get("PRIMARY_PAIR")
    if not isinstance(primary, Mapping):
        raise BoreasStage2StorageOptimizationError("Stage-1 primary pair is absent")
    if (
        primary.get("map_sequence_id"),
        primary.get("query_sequence_id"),
    ) != (PRIMARY_MAP_SEQUENCE, PRIMARY_QUERY_SEQUENCE):
        raise BoreasStage2StorageOptimizationError("Stage-1 primary pair changed")

    allowlist = root / "boreas_v2_stage2_download_allowlist.csv"
    role_count = {"TARGET_MAP": 0, "QUERY": 0}
    role_bytes = {"TARGET_MAP": 0, "QUERY": 0}
    max_object_bytes = 0
    query_sizes: list[int] = []
    with allowlist.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = {
            "selection_role",
            "sequence_id",
            "key",
            "timestamp_us",
            "last_modified",
            "size_bytes",
            "selection_reason",
        }
        if set(reader.fieldnames or ()) != expected_fields:
            raise BoreasStage2StorageOptimizationError("Stage-1 allowlist schema changed")
        seen_keys: set[str] = set()
        for row in reader:
            role = row["selection_role"]
            if role not in role_count:
                raise BoreasStage2StorageOptimizationError("unexpected Stage-1 allowlist role")
            key = row["key"]
            if key in seen_keys or re.fullmatch(r"[^/]+/lidar/[0-9]+\.bin", key) is None:
                raise BoreasStage2StorageOptimizationError("unsafe/duplicate Stage-1 allowlist key")
            sequence = row["sequence_id"]
            expected_sequence = (
                PRIMARY_MAP_SEQUENCE if role == "TARGET_MAP" else PRIMARY_QUERY_SEQUENCE
            )
            if sequence != expected_sequence or not key.startswith(f"{sequence}/lidar/"):
                raise BoreasStage2StorageOptimizationError("Stage-1 allowlist pair binding changed")
            size = int(row["size_bytes"])
            if size <= 0:
                raise BoreasStage2StorageOptimizationError("invalid Stage-1 object size")
            seen_keys.add(key)
            role_count[role] += 1
            role_bytes[role] += size
            max_object_bytes = max(max_object_bytes, size)
            if role == "QUERY":
                query_sizes.append(size)
    if role_count != {"TARGET_MAP": EXPECTED_MAP_OBJECT_COUNT, "QUERY": EXPECTED_QUERY_OBJECT_COUNT}:
        raise BoreasStage2StorageOptimizationError("Stage-1 allowlist object counts changed")
    if role_bytes != {"TARGET_MAP": EXPECTED_MAP_REMOTE_BYTES, "QUERY": EXPECTED_QUERY_REMOTE_BYTES}:
        raise BoreasStage2StorageOptimizationError("Stage-1 allowlist remote bytes changed")
    if sum(role_count.values()) != EXPECTED_ALLOWLIST_OBJECT_COUNT:
        raise BoreasStage2StorageOptimizationError("Stage-1 allowlist total count changed")
    if sum(role_bytes.values()) != EXPECTED_REMOTE_BYTES:
        raise BoreasStage2StorageOptimizationError("Stage-1 allowlist total bytes changed")

    plan = _load_json(root / "boreas_v2_stage2_download_plan.json")
    if (
        plan.get("estimated_download_object_count", EXPECTED_ALLOWLIST_OBJECT_COUNT)
        != EXPECTED_ALLOWLIST_OBJECT_COUNT
        or plan.get("estimated_download_bytes") != EXPECTED_REMOTE_BYTES
        or plan.get("primary_pair")
        != {
            "map_sequence_id": PRIMARY_MAP_SEQUENCE,
            "query_sequence_id": PRIMARY_QUERY_SEQUENCE,
        }
    ):
        raise BoreasStage2StorageOptimizationError("Stage-1 download plan changed")
    no_icp = _load_json(root / "NO_ICP_ATTESTATION.json")
    no_lidar = _load_json(root / "NO_LIDAR_PAYLOAD_ATTESTATION.json")
    for field in (
        "open3d_registration_call_count",
        "pcl_cli_invocation_count",
        "other_registration_process_count",
        "registration_execution_count",
    ):
        if no_icp.get(field) != 0:
            raise BoreasStage2StorageOptimizationError(f"Stage-1 {field} is nonzero")
    for field in (
        "downloaded_lidar_bytes",
        "downloaded_lidar_object_count",
        "downloaded_lidar_payload_count",
    ):
        if no_lidar.get(field) != 0:
            raise BoreasStage2StorageOptimizationError(f"Stage-1 {field} is nonzero")

    return {
        "allowlist_object_count": EXPECTED_ALLOWLIST_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "allowlist_sha256": STAGE1_ALLOWLIST_SHA256,
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
        "frozen_manifest_file_sha256": STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "manifest_root_sha256": STAGE1_MANIFEST_ROOT_SHA256,
        "map_object_count": role_count["TARGET_MAP"],
        "map_remote_bytes": role_bytes["TARGET_MAP"],
        "max_object_bytes": max_object_bytes,
        "no_icp_attestation_sha256": STAGE1_NO_ICP_SHA256,
        "no_lidar_attestation_sha256": STAGE1_NO_LIDAR_SHA256,
        "pair_selection_sha256": STAGE1_PAIR_SELECTION_SHA256,
        "primary_pair": {
            "map_sequence_id": PRIMARY_MAP_SEQUENCE,
            "query_sequence_id": PRIMARY_QUERY_SEQUENCE,
        },
        "query_object_count": role_count["QUERY"],
        "query_remote_bytes": role_bytes["QUERY"],
        "signed_file_count": len(signed_rows),
        "top_100_query_remote_bytes": sum(sorted(query_sizes, reverse=True)[:100]),
        "verification_pass": True,
    }


def inspect_repository_gate(repository: str | Path, *, require_clean: bool) -> dict[str, Any]:
    repository_path = Path(repository).resolve(strict=True)
    branch = _git(repository_path, "branch", "--show-current")
    head = _git(repository_path, "rev-parse", "HEAD")
    status = _git(repository_path, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != EXPECTED_BRANCH:
        raise BoreasStage2StorageOptimizationError(
            f"formal storage audit requires branch {EXPECTED_BRANCH}"
        )
    if require_clean and status:
        raise BoreasStage2StorageOptimizationError("formal storage audit requires clean worktree")
    tag_rows: dict[str, str] = {}
    for tag, expected in EXPECTED_TAG_COMMITS.items():
        actual = _git(repository_path, "rev-parse", f"refs/tags/{tag}^{{commit}}")
        if actual != expected:
            raise BoreasStage2StorageOptimizationError(f"protected tag moved: {tag}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", actual, head], cwd=repository_path
        )
        if ancestor.returncode != 0:
            raise BoreasStage2StorageOptimizationError(f"protected tag not in HEAD history: {tag}")
        tag_rows[tag] = actual
    return {
        "branch": branch,
        "head": head,
        "protected_tags": tag_rows,
        "worktree_clean": not bool(status),
    }


def _assert_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in REQUIRED_ENVIRONMENT}
    missing = [name for name, value in values.items() if value != "1"]
    if missing:
        raise BoreasStage2StorageOptimizationError(
            "required fail-closed environment is absent: " + ", ".join(missing)
        )
    return values


def _assert_no_local_lidar_payload(data_root: Path) -> list[str]:
    if not data_root.is_dir() or data_root.is_symlink():
        raise BoreasStage2StorageOptimizationError("Boreas Stage-1 data root is absent or unsafe")
    found = sorted(
        str(path)
        for path in data_root.rglob("*.bin")
        if path.is_file() and path.parent.name == "lidar"
    )
    if found:
        raise BoreasStage2StorageOptimizationError("Boreas lidar payload already exists locally")
    return found


def _sha256sums_payload(root: Path, names: Iterable[str]) -> bytes:
    return "".join(
        f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)
    ).encode("utf-8")


def _freeze_runtime_assets(runtime_root: Path, frozen_root: Path) -> None:
    runtime = runtime_root.resolve(strict=True)
    destination = frozen_root.resolve(strict=False)
    if destination != frozen_root or frozen_root.exists() or frozen_root.is_symlink():
        raise BoreasStage2StorageOptimizationError("fresh canonical absent frozen root required")
    if not (runtime / "SHA256SUMS").is_file():
        raise BoreasStage2StorageOptimizationError("runtime closure is incomplete")
    rows = _verify_sha256sums(runtime)
    expected_names = sorted(
        path.name
        for path in runtime.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    if sorted(row["path"] for row in rows) != expected_names:
        raise BoreasStage2StorageOptimizationError("runtime SHA256SUMS closure is not exact")
    manifest = _load_json(runtime / "frozen_manifest.json")
    claimed_root = manifest.pop("manifest_root_sha256", None)
    if claimed_root != compact_sha256(manifest):
        raise BoreasStage2StorageOptimizationError("runtime manifest self-hash mismatch")
    shutil.copytree(runtime, destination, copy_function=shutil.copy2)


def _reasoned_original_breakdown(old: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Disaggregate only what the legacy formula actually states.

    Zero rows are intentional: they distinguish an absent explicit estimate
    from an inference that the legacy coarse envelopes may have covered it.
    """

    raw = int(old["download_bytes"])
    decoded = int(old["decoded_or_unpacked_working_bytes"])
    map_total = int(old["target_map_bytes"])
    canonical = int(old["canonical_bundle_bytes"])
    temporary = int(old["temporary_processing_bytes"])
    subtotal = int(old["subtotal_before_safety_bytes"])
    total = int(old["required_safe_disk_bytes"])
    map_base = int(old["target_map_download_bytes"])
    if (
        raw != EXPECTED_REMOTE_BYTES
        or decoded != 2 * raw
        or map_total != 2 * map_base
        or canonical != 2 * raw
        or temporary != raw
        or subtotal != raw + decoded + map_total + canonical + temporary
        or total != (3 * subtotal + 1) // 2
    ):
        raise BoreasStage2StorageOptimizationError("legacy disk-budget arithmetic changed")

    def row(
        component: str,
        *,
        raw_bytes: int,
        duplication: float,
        lifetime: str,
        retained: bool,
        reason: str,
        source: str,
        unit_count: int | None = None,
        bytes_per_unit: int | None = None,
        contribution: int | None = None,
    ) -> dict[str, Any]:
        return {
            "bytes_per_unit": bytes_per_unit,
            "component": component,
            "duplication_factor": duplication,
            "lifetime": lifetime,
            "peak_contribution_bytes": (
                int(raw_bytes * duplication) if contribution is None else contribution
            ),
            "raw_bytes": raw_bytes,
            "reason": reason,
            "retained_after_stage2": retained,
            "simultaneously_live": True if raw_bytes or contribution else False,
            "source_of_estimate": source,
            "unit_count": unit_count,
        }

    rows = [
        row(
            "raw lidar payload",
            raw_bytes=raw,
            duplication=1.0,
            lifetime="PERSISTENT_RAW_IN_LEGACY_ENVELOPE",
            retained=True,
            reason="Legacy download_bytes treated the complete 20,061-object allowlist as local.",
            source="boreas_v2_stage2_disk_budget.json:download_bytes",
            unit_count=EXPECTED_ALLOWLIST_OBJECT_COUNT,
        ),
        row(
            "temporary download files",
            raw_bytes=temporary,
            duplication=1.0,
            lifetime="TEMPORARY_DOWNLOAD_COARSE_ENVELOPE",
            retained=False,
            reason="Legacy temporary_processing_bytes equals the complete remote payload.",
            source="boreas_v2_stage2_disk_budget.json:temporary_processing_bytes",
            unit_count=EXPECTED_ALLOWLIST_OBJECT_COUNT,
        ),
        row(
            "decoded arrays",
            raw_bytes=raw,
            duplication=2.0,
            lifetime="TEMPORARY_DECODED_COARSE_ENVELOPE",
            retained=False,
            reason="Legacy decoded_or_unpacked_working_bytes is exactly two complete raw payloads.",
            source="boreas_v2_stage2_disk_budget.json:decoded_or_unpacked_working_bytes",
            unit_count=EXPECTED_ALLOWLIST_OBJECT_COUNT,
            contribution=decoded,
        ),
        row(
            "map scan materialization",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_SEPARATELY_MODELLED",
            retained=False,
            reason="No separate legacy field; it may be covered by decoded_working but cannot be proved.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "accumulated target map",
            raw_bytes=map_base,
            duplication=1.0,
            lifetime="PERSISTENT_MAP_COARSE_ENVELOPE",
            retained=True,
            reason="One of the two map-sized copies encoded by target_map_bytes.",
            source="target_map_bytes == 2 * target_map_download_bytes",
            unit_count=1,
            bytes_per_unit=map_base,
        ),
        row(
            "voxel map copies",
            raw_bytes=map_base,
            duplication=1.0,
            lifetime="TEMPORARY_OR_PERSISTENT_MAP_COPY_COARSE_ENVELOPE",
            retained=False,
            reason="Second map-sized copy encoded by the legacy twofold target-map envelope.",
            source="target_map_bytes == 2 * target_map_download_bytes",
            unit_count=1,
            bytes_per_unit=map_base,
        ),
        row(
            "query decoded points",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_SEPARATELY_MODELLED",
            retained=False,
            reason="No separate legacy field; query decode is only implicit in decoded_working.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "canonical source arrays",
            raw_bytes=raw,
            duplication=2.0,
            lifetime="PERSISTENT_CANONICAL_COARSE_ENVELOPE",
            retained=True,
            reason="Accounting placement for the indivisible legacy canonical_bundle; it does not prove that every byte is a source array.",
            source="boreas_v2_stage2_disk_budget.json:canonical_bundle_bytes",
            unit_count=PLANNED_SNAPSHOT_COUNT,
            contribution=canonical,
        ),
        row(
            "canonical target arrays",
            raw_bytes=0,
            duplication=0.0,
            lifetime="SUBSUMED_NOT_DISAGGREGATABLE",
            retained=True,
            reason="Legacy canonical_bundle did not split source and target bytes.",
            source="legacy canonical_bundle is aggregate only",
        ),
        row(
            "Open3D copies",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=False,
            reason="No evidence in the legacy JSON of a separately retained Open3D copy.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "PCL copies",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=False,
            reason="No evidence in the legacy JSON of a separately retained PCL copy.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "per-snapshot target copies",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=False,
            reason="The legacy JSON does not encode 100 physical target copies.",
            source="absence of a dedicated legacy field",
            unit_count=PLANNED_SNAPSHOT_COUNT,
        ),
        row(
            "intermediate PCD/bin files",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=False,
            reason="No separate legacy estimate.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "geometry metrics",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=True,
            reason="Small evidence was not separately estimated.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "publication artifacts",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=True,
            reason="Not part of the legacy disk formula.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "resume/checkpoint files",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=True,
            reason="Not part of the legacy disk formula.",
            source="absence of a dedicated legacy field",
        ),
        row(
            "safety factor",
            raw_bytes=subtotal,
            duplication=0.5,
            lifetime="CAPACITY_RESERVE",
            retained=False,
            reason="The legacy 1.5 multiplier adds 50% after all coarse components were summed as simultaneous.",
            source="minimum_safety_multiplier=1.5",
            contribution=total - subtotal,
        ),
        row(
            "filesystem overhead",
            raw_bytes=0,
            duplication=0.0,
            lifetime="NOT_EXPLICITLY_MODELLED",
            retained=False,
            reason="No separate filesystem-overhead field; only the undifferentiated 50% safety factor exists.",
            source="absence of a dedicated legacy field",
        ),
    ]
    if sum(int(value["peak_contribution_bytes"]) for value in rows) != total:
        raise BoreasStage2StorageOptimizationError("original breakdown does not close")
    return rows


def _lifecycle_rows() -> list[dict[str, Any]]:
    rows = [
        ("map S3 LiDAR object", "before map processing", "REMOTE_ONLY", "S3", False, "key,size,ETag,LastModified", "map stream", "never a frozen local file", "identity mismatch fails", "remote provenance"),
        ("map raw scan", "one-object materialization", "TEMPORARY_DOWNLOAD", "tmp_download", False, "remote identity,local SHA", "decoder", "unlink immediately after authenticated consumption", "partial removed then reacquired", "receipt is evidence; payload is not"),
        ("map decoded XYZ", "transform/accumulate", "TEMPORARY_DECODED", "tmp_decode or memory", False, "source object SHA,decode contract SHA", "incremental map builder", "discard after state transition commits", "orphan decode rejected", "none"),
        ("target-map accumulator checkpoint", "map streaming", "CHECKPOINT", "checkpoints", True, "prior/new state SHA,object identity,contract SHA", "resume", "retain until final map independently verifies", "authenticated chain only", "resume evidence"),
        ("canonical target map", "after map finalization", "PERSISTENT_MAP", "target_map/<sha256>", True, "content SHA,scientific contract SHA", "query metrics and both backends", "content-addressed immutable single copy", "rebuild or verify by SHA", "shared canonical target"),
        ("query S3 LiDAR object", "before query screening", "REMOTE_ONLY", "S3", False, "key,size,ETag,LastModified", "query pass 1 or selected pass 2", "never a frozen local raw default", "identity mismatch fails", "remote provenance"),
        ("query pass-1 raw scan", "blind geometry screening", "TEMPORARY_DOWNLOAD", "tmp_download", False, "remote identity,local SHA", "decoder", "unlink after metric/checkpoint commit", "partial removed then reacquired", "receipt is evidence; payload is not"),
        ("query pass-1 decoded XYZ", "blind geometry screening", "TEMPORARY_DECODED", "tmp_decode or memory", False, "source object SHA,decode contract SHA", "geometry-only metrics", "discard after metric/checkpoint commit", "orphan decode rejected", "none"),
        ("query geometry row", "blind geometry screening", "PERSISTENT_RESULT", "geometry_metrics", True, "object identity,GT/calibration/target/row SHA", "frozen selector and verifier", "immutable append/final canonical table", "authenticated row skips reprocessing", "R07/R08/R14 input"),
        ("query receipt", "both query passes", "PERSISTENT_RESULT", "receipts", True, "key,size,ETag,LastModified,local SHA", "independent verifier", "retain", "identity mismatch fails", "download provenance"),
        ("selected canonical source", "pass 2 after blind freeze", "PERSISTENT_CANONICAL", "selected_sources/<sha256>", True, "selection SHA,source SHA,decode contract SHA", "both backends", "content-addressed immutable", "regenerate only from same remote identity", "shared canonical source"),
        ("snapshot manifest", "after R14 freeze", "PERSISTENT_RESULT", "manifests", True, "canonical source SHA,target map SHA,selection SHA", "both backends and verifier", "retain; target referenced, never copied", "immutable", "R14 freeze"),
        ("PCL converted input", "one backend invocation", "TEMPORARY_DECODED", "tmp_pcl", False, "canonical parent SHA,conversion contract SHA", "PCL CLI", "unlink in finally block", "rederive deterministically", "not an evidence master"),
        ("backend result", "future authorized execution only", "PERSISTENT_RESULT", "results", True, "snapshot/backend/parameter/input SHA", "analysis/verifier", "retain", "future execution checkpoint", "future result evidence"),
    ]
    return [dict(zip(LIFECYCLE_FIELDS, row, strict=True)) for row in rows]


def _original_explanation(old: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    subtotal = int(old["subtotal_before_safety_bytes"])
    total = int(old["required_safe_disk_bytes"])
    explicit = [
        value
        for value in rows
        if int(value["peak_contribution_bytes"]) > 0
        and value["component"] != "safety factor"
    ]
    lines = [
        "# Boreas v2 Stage-2 原始磁盘预算逐项审计",
        "",
        "## 结论",
        "",
        "旧预算的 990.390954 GiB 是一个粗粒度的同时存活上界，不是不可避免的物理需求。"
        "它先把完整 raw、两倍 decoded、两倍 map、两倍 canonical 与一个完整 raw 大小的"
        " temporary 全部相加，再统一增加 50% 容量。旧 JSON 没有对象生命周期，也没有证明"
        "这些大项会在同一时刻全部存在。",
        "",
        "旧文件也没有单列 Open3D/PCL 副本或 100 份 per-snapshot target；因此本审计不能声称"
        "旧实现实际保存了这些副本。它们至多可能被 decoded/canonical/map 的粗粒度 envelope"
        "隐含覆盖。优化的依据是明确生命周期、单 target 内容寻址、共享 backend 输入与流式临时"
        "消费，而不是把未证明的旧副本当成事实。",
        "",
        "## 原公式",
        "",
        "```text",
        "subtotal = download + decoded_working + target_map + canonical_bundle + temporary_processing",
        f"         = {subtotal} bytes",
        "required = ceil(subtotal × 1.5)",
        f"         = {total} bytes = {total / (1024**3):.6f} GiB",
        "```",
        "",
        "## 有非零显式贡献的项目",
        "",
        "| component | peak bytes | legacy evidence |",
        "|---|---:|---|",
    ]
    for value in explicit:
        lines.append(
            f"| {value['component']} | {int(value['peak_contribution_bytes'])} | "
            f"{value['source_of_estimate']} |"
        )
    safety = next(value for value in rows if value["component"] == "safety factor")
    lines.extend(
        [
            f"| safety factor | {int(safety['peak_contribution_bytes'])} | "
            "minimum_safety_multiplier=1.5 |",
            "",
            "完整 CSV/JSON 还保留所有要求审计的零贡献行；零表示旧模型没有单独估计，不能解释成"
            "真实运行一定为零。",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_bundle_contract() -> dict[str, Any]:
    source_reference = "REFERENCED_CONTENT_ADDRESSED_CANONICAL_SOURCE_SHA256"
    target_reference = "REFERENCED_CONTENT_ADDRESSED_TARGET_MAP_SHA256"
    return {
        "actual_canonical_source_count": 0,
        "actual_target_map_count": 0,
        "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
        "canonical_array_container": "NPY_V1_CONTENT_ADDRESSED_IMMUTABLE",
        "future_open3d_source_sha256": source_reference,
        "future_open3d_target_sha256": target_reference,
        "future_pcl_source_sha256": source_reference,
        "future_pcl_target_sha256": target_reference,
        "input_equality_requirements": {
            "future_open3d_source_sha256_equals_future_pcl_source_sha256": True,
            "future_open3d_target_sha256_equals_future_pcl_target_sha256": True,
        },
        "canonicalizer_point_count_contract": PREPROCESSING_UNRESOLVED,
        "pcl_runtime_conversion": {
            "cleanup": "DELETE_IN_FINALLY_AFTER_RESULT_OR_ERROR",
            "determinism": "BYTE_IDENTICAL_FROM_SAME_CANONICAL_NPY_AND_CONVERSION_CONTRACT",
            "directory": "tmp_pcl",
            "persistent": False,
            "reverse_validation": (
                "TEMPORARY_PCD_REPARSE_MUST_EQUAL_THE_DETERMINISTIC_FLOAT32_PROJECTION_"
                "OF_THE_CANONICAL_ARRAY"
            ),
        },
        "prohibitions": [
            "NO_PERSISTENT_OPEN3D_ONLY_SOURCE_COPY",
            "NO_PERSISTENT_PCL_ONLY_SOURCE_COPY",
            "NO_PERSISTENT_OPEN3D_ONLY_TARGET_COPY",
            "NO_PERSISTENT_PCL_ONLY_TARGET_COPY",
            "NO_PER_SNAPSHOT_TARGET_MAP_COPY",
        ],
        "role": "FUTURE_STAGE2_STORAGE_CONTRACT_NOT_EXECUTION_AUTHORIZATION",
        "schema_version": "boreas_v2_canonical_bundle_storage_contract_v2",
        "shared_input_architecture": "ONE_CANONICAL_SOURCE_AND_TARGET_FAN_OUT_TO_BOTH_BACKENDS",
    }


def _streaming_map_contract(stage1: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actual_map_object_processing_count": 0,
        "actual_target_map_count": 0,
        "algorithm": {
            "accumulator": "VOXEL_KEY_TO_COUNT_AND_FLOAT64_SUM_XYZ",
            "duplicate_handling": "KEEP_ALL_FINITE_POINTS_IN_SUFFICIENT_STATISTICS",
            "final_point_order": "LEXICOGRAPHIC_VOXEL_KEY_X_Y_Z",
            "input_order": "FROZEN_ALLOWLIST_ORDER_WITH_EXPLICIT_ORDINAL",
            "numeric_dtype": "LITTLE_ENDIAN_FLOAT64",
            "processing_order": "SCAN_ORDINAL_THEN_POINT_ORDINAL",
            "representative": "CENTROID_FROM_FIXED_ORDER_FLOAT64_SUM",
            "worker_count_effect": "NONE_WORKERS_MAY_NOT_CHANGE_REDUCTION_ORDER",
        },
        "allowlist_object_count": stage1["map_object_count"],
        "allowlist_remote_bytes": stage1["map_remote_bytes"],
        "architecture_test_scope": "SYNTHETIC_TINY_POINT_CLOUDS_ONLY",
        "batch_incremental_semantics": "BYTE_EXACT_FOR_SAME_ORDERED_INPUTS_AND_RULE",
        "content_addressing": {
            "planned_physical_target_map_copy_count": 1,
            "planned_snapshot_target_reference_count": PLANNED_SNAPSHOT_COUNT,
            "planned_unique_target_map_count": 1,
            "snapshot_target_storage": "SHA256_REFERENCE_ONLY",
        },
        "crash_resume_semantics": "AUTHENTICATED_STATE_CHAIN_PRODUCES_SAME_FINAL_BYTES",
        "production_disk_checkpoint": {
            "budget_upper_bound_bytes": stage1["map_remote_bytes"] + 128,
            "format": "ONE_PREALLOCATED_LITTLE_ENDIAN_FLOAT64_XYZ_REPLAY_ARRAY",
            "identity": (
                "Frozen allowlist sizes determine the complete point count; an append ledger "
                "authenticates each fixed-order scan byte range and map-state transition."
            ),
            "reason": (
                "On crash, replay the authenticated transformed XYZ prefix locally to rebuild "
                "the deterministic accumulator; completed remote objects are not downloaded again."
            ),
            "replay_array_payload_bytes": stage1["map_remote_bytes"],
            "ledger_and_receipt_overhead_budget_component": (
                "SMALL_EVIDENCE_UPPER_BOUND_NOT_THE_128_BYTE_ARRAY_ALLOWANCE"
            ),
            "retention": "DELETE_ONLY_AFTER_FINAL_CONTENT_ADDRESSED_TARGET_COMMITS_AND_VERIFIES",
            "synthetic_json_checkpoint_scope": "TINY_FIXTURE_ONLY_NOT_PRODUCTION_DISK_BUDGET",
            "target_npy_byte_exact_after_authenticated_replay": True,
            "internal_state_sha_semantics": (
                "REPRESENTATION_SPECIFIC_NOT_COMPARED_BETWEEN_DIRECT_TRANSFORM_AND_"
                "PRETRANSFORMED_REPLAY"
            ),
        },
        "flow": [
            "REMOTE_ONLY",
            "TEMPORARY_DOWNLOAD",
            "TEMPORARY_DECODED",
            "FROZEN_GT_AND_CALIBRATION_TRANSFORM",
            "INCREMENTAL_ACCUMULATOR",
            "AUTHENTICATED_CHECKPOINT",
            "DELETE_TEMPORARIES",
            "ONE_PERSISTENT_CONTENT_ADDRESSED_MAP",
        ],
        "floating_point_risk": (
            "Floating-point addition is non-associative; byte equality would be false if worker "
            "or resume partition changed reduction order. The architecture therefore fixes scan "
            "and point order and does not parallel-reduce voxel sums."
        ),
        "local_subset_policy": {
            "default": "NO_LOCAL_CROP_ONE_GLOBAL_TARGET",
            "future_exception": (
                "Only a separately preregistered scientific contract before any registration may "
                "define a deterministic subset from the same frozen global target."
            ),
            "registration_result_adaptation": "FORBIDDEN",
            "subset_content_addressed_and_deduplicated": True,
        },
        "primary_map_sequence_id": PRIMARY_MAP_SEQUENCE,
        "real_boreas_map_built": False,
        "role": "DRY_RUN_ARCHITECTURE_ONLY",
        "schema_version": "boreas_v2_stage2_streaming_map_contract_v1",
        "scientific_preprocessing_parameters": {
            "association_distance": PREPROCESSING_UNRESOLVED,
            "normal_neighborhood": PREPROCESSING_UNRESOLVED,
            "range_limits": PREPROCESSING_UNRESOLVED,
            "voxel_origin_xyz_m": PREPROCESSING_UNRESOLVED,
            "voxel_size_m": PREPROCESSING_UNRESOLVED,
        },
        "storage_planner_parameter_authority": "NONE",
    }


def _streaming_query_contract(stage1: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actual_geometry_metric_row_count": 0,
        "actual_query_object_processing_count": 0,
        "actual_selected_source_count": 0,
        "allowlist_object_count": stage1["query_object_count"],
        "allowlist_remote_bytes": stage1["query_remote_bytes"],
        "architecture_test_scope": "SYNTHETIC_TINY_POINT_CLOUDS_ONLY",
        "first_pass": {
            "blind_to_registration_results": True,
            "candidate_object_count": stage1["query_object_count"],
            "persistent_fields": [
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
            "raw_payload_cleanup": "IMMEDIATE_AFTER_METRIC_AND_CHECKPOINT_COMMIT",
        },
        "primary_query_sequence_id": PRIMARY_QUERY_SEQUENCE,
        "real_boreas_geometry_computed": False,
        "role": "DRY_RUN_ARCHITECTURE_ONLY",
        "schema_version": "boreas_v2_stage2_streaming_query_contract_v1",
        "second_pass": {
            "begins_only_after_R14_selection_freeze": True,
            "canonical_source_count": PLANNED_SNAPSHOT_COUNT,
            "checkpoint_file": "processed_selected_sources.jsonl",
            "checkpoint_record_kind": "SELECTED_SOURCE",
            "checkpoint_processing_binding": (
                "SHA256_OF_CANONICALIZATION_CONTRACT_AND_FROZEN_SELECTION_RECORD"
            ),
            "object_identity_must_match_first_pass": True,
            "raw_payload_cleanup": "IMMEDIATE_AFTER_CANONICAL_SOURCE_COMMIT",
            "resume": "SKIP_ONLY_AUTHENTICATED_COMPLETED_SELECTION_PREFIX",
            "selected_object_count": PLANNED_SNAPSHOT_COUNT,
        },
        "selection_contract": {
            "geometry_only_selector": "phase_a_harness.real_data_preparation.selection",
            "planned_rich_snapshot_count": PLANNED_RICH_SNAPSHOT_COUNT,
            "planned_snapshot_count": PLANNED_SNAPSHOT_COUNT,
            "planned_weak_snapshot_count": PLANNED_WEAK_SNAPSHOT_COUNT,
            "registration_derived_fields_forbidden": True,
            "storage_planner_selects_snapshots": False,
        },
        "two_pass_adjudication": {
            "preferred": True,
            "reproducibility": "PASS_REMOTE_IDENTITY_AND_ALL_CONTRACT_SHA_VALUES_ARE_FROZEN",
            "R07": "PASS_SELECTION_USES_ONLY_PREREGISTERED_GEOMETRY_METRICS",
            "R08": "PASS_BOTH_BACKENDS_LATER_READ_THE_SAME_CONTENT_ADDRESSED_INPUTS",
            "R14": "PASS_SELECTION_RECORD_IS_FROZEN_BEFORE_REGISTRATION",
            "sha_auditability": "PASS_RECEIPTS_BIND_BOTH_PASSES_TO_KEY_SIZE_ETAG_LASTMODIFIED_AND_SHA",
        },
    }


def _checkpoint_contract() -> dict[str, Any]:
    fields = [
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
    ]
    return {
        "actual_completed_map_object_count": 0,
        "actual_completed_query_object_count": 0,
        "actual_completed_selected_source_count": 0,
        "append_semantics": "LOCKED_O_APPEND_CANONICAL_HASH_CHAIN_FILE_AND_DIRECTORY_FSYNC",
        "checkpoint_files": [
            "processed_map_objects.jsonl",
            "processed_query_objects.jsonl",
            "processed_selected_sources.jsonl",
        ],
        "completed_record_fields": fields,
        "identity_change_semantics": "FAIL_ON_SIZE_ETAG_LASTMODIFIED_OR_CONTRACT_CHANGE",
        "manifest_overwrite": "FORBIDDEN",
        "orphan_state": "REJECT_NEVER_AUTO_ADOPT",
        "partial_temp": "DELETE_THEN_REACQUIRE_SAME_FROZEN_OBJECT_IDENTITY",
        "production_map_resume_payload": (
            "ONE_PREALLOCATED_FLOAT64_XYZ_REPLAY_ARRAY_WITH_AUTHENTICATED_FIXED_ORDER_BYTE_RANGES"
        ),
        "resume": "SKIP_ONLY_COMPLETE_AUTHENTICATED_CHAIN_MEMBERS",
        "schema_version": "boreas_v2_stage2_checkpoint_contract_v1",
        "strict_resume_implemented": True,
    }


def _temp_contract() -> dict[str, Any]:
    return {
        "directories": {
            "tmp_decode": "TEMPORARY_DECODED",
            "tmp_download": "TEMPORARY_DOWNLOAD",
            "tmp_pcl": "TEMPORARY_DECODED_BACKEND_CONVERSION",
        },
        "frozen_manifest_inclusion": False,
        "persistent_directories": [
            "receipts",
            "target_map",
            "geometry_metrics",
            "selected_sources",
            "manifests",
            "results",
            "checkpoints",
        ],
        "rules": [
            "EACH_TEMP_ROOT_HAS_EXPLICIT_MARKER_AND_PURPOSE",
            "NO_SYMLINK_OR_UNKNOWN_ENTRY_CLEANUP",
            "SUCCESSFUL_CONSUMPTION_DELETES_TEMP_IMMEDIATELY",
            "CRASH_RESUME_REMOVES_ONLY_MARKED_PARTIAL_TEMP_FILES",
            "TEMP_IS_NEVER_THE_ONLY_COPY_OF_SCIENTIFIC_EVIDENCE",
            "PERSISTENT_OUTPUT_MUST_COMMIT_BEFORE_TEMP_DELETE",
        ],
        "safe_to_clear_when_validated": True,
        "schema_version": "boreas_v2_stage2_temp_file_contract_v1",
    }


def _execution_modes_contract(budget: Mapping[str, Any]) -> dict[str, Any]:
    science_fingerprint = compact_sha256(
        {
            "allowlist_sha256": STAGE1_ALLOWLIST_SHA256,
            "backend_parameter_contract_sha256": BACKEND_PARAMETER_CONTRACT_SHA256,
            "primary_pair": [PRIMARY_MAP_SEQUENCE, PRIMARY_QUERY_SEQUENCE],
            "planned_rich": PLANNED_RICH_SNAPSHOT_COUNT,
            "planned_snapshots": PLANNED_SNAPSHOT_COUNT,
            "planned_weak": PLANNED_WEAK_SNAPSHOT_COUNT,
            "preprocessing": PREPROCESSING_UNRESOLVED,
        }
    )
    common = {
        "canonical_source_semantics": "IDENTICAL",
        "geometry_metric_semantics": "IDENTICAL",
        "science_contract_fingerprint": science_fingerprint,
        "selection_semantics": "IDENTICAL",
        "target_map_semantics": "IDENTICAL",
    }
    return {
        "actual_execution_mode": None,
        "architecture_only": True,
        "modes": {
            "FULL_RAW_CACHE": {
                **common,
                "budget_mode": "CONSERVATIVE_FULL_CACHE",
                "raw_policy": "PERSIST_ALL_20061_AUTHENTICATED_RAW_OBJECTS",
                "target_policy": "ONE_CONTENT_ADDRESSED_TARGET",
            },
            "STREAMING_LOW_DISK": {
                **common,
                "budget_mode": "RECOMMENDED_OPERATIONAL",
                "raw_policy": "TEMPORARY_ONE_OBJECT_PROCESS_RECEIPT_DELETE",
                "target_policy": "ONE_CONTENT_ADDRESSED_TARGET",
            },
        },
        "mode_count": 2,
        "mode_scientific_equivalence_required": True,
        "optimized_budget_contract_sha256": compact_sha256(budget),
        "real_execution_performed": False,
        "schema_version": "boreas_v2_stage2_execution_modes_v1",
        "synthetic_fixture_equivalence": "PASS",
    }


def _budget_csv_rows(budget: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
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
    return [{name: value[name] for name in fields} for value in budget["modes"]]


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


def _summary_answers(
    *,
    budget: Mapping[str, Any],
    test_counts: Mapping[str, int],
    verifier_command: str,
) -> list[dict[str, Any]]:
    modes = {row["mode"]: row for row in budget["modes"]}
    minimum = modes["THEORETICAL_MINIMUM"]
    recommended = modes["RECOMMENDED_OPERATIONAL"]
    full = modes["CONSERVATIVE_FULL_CACHE"]
    current = int(recommended["current_free_bytes"])
    answers: list[str] = [
        (
            "旧公式把完整 raw、2×decoded、2×map、2×canonical 和一个完整 raw 大小的 temp "
            "同时相加为 708,949,459,392 bytes，再乘 1.5 得 1,063,424,189,088 bytes "
            "（990.390954 GiB）。"
        ),
        "最大显式重复包络是 decoded 与 canonical，各为完整 remote payload 的 2 倍；随后又叠加 50% 总体余量。",
        "没有证据表明旧设计实际存在 100 份 target map；旧 JSON 没有单列这一项。新合同明确禁止并检测这种复制。",
        "没有证据表明旧设计实际持久保存 Open3D/PCL 双份输入；新合同让两个 backend 引用同一 canonical SHA。",
        "优化后保存 1 份 content-addressed 全局 target map；100 个未来 snapshot 仅引用其 SHA。",
        "可以。候选 query 第一遍只保留 authenticated receipt、checkpoint 和 geometry-only row，成功提交后删除 raw/decoded temp。",
        "是。第一遍 blind geometry screening 后冻结 100 个选择；第二遍只为这 100 个重取并生成 canonical source。",
        "每个 selected source 以 deterministic canonical NPY 内容寻址保存一次，并由 Open3D/PCL 共享；raw temp 随后删除。",
        "不必须。STREAMING_LOW_DISK 不永久保存 104,158,637,472 remote bytes；FULL_RAW_CACHE 只是可选操作模式。",
        f"THEORETICAL_MINIMUM 要求 {minimum['recommended_free_disk_bytes']} bytes（{minimum['recommended_free_disk_GiB']:.6f} GiB）。",
        f"RECOMMENDED_OPERATIONAL 要求 {recommended['recommended_free_disk_bytes']} bytes（{recommended['recommended_free_disk_GiB']:.6f} GiB）。",
        f"CONSERVATIVE_FULL_CACHE 要求 {full['recommended_free_disk_bytes']} bytes（{full['recommended_free_disk_GiB']:.6f} GiB）。",
        (
            f"当前 {current} bytes（{current / GIB:.6f} GiB）"
            f"{'足够' if budget['CURRENT_DISK_SUFFICIENT'] else '不足'}运行推荐的 STREAMING_LOW_DISK 模式；"
            f"它{'不' if not full['CURRENT_DISK_SUFFICIENT'] else ''}满足 FULL_RAW_CACHE。"
        ),
        (
            f"推荐流式模式额外需要 {recommended['additional_GiB_required']:.6f} GiB；"
            f"full-cache 额外需要 {full['additional_GiB_required']:.6f} GiB。"
        ),
        (
            "真实 target map 大小尚未知；不设 voxel 参数的保守持久上界是 "
            f"{budget['size_estimates']['target_map_upper_bound_bytes']} bytes。"
        ),
        (
            "推荐模式最大 mutually-exclusive temporary phase 是 map checkpoint，上界 "
            f"{recommended['temporary_peak_bytes']} bytes。"
        ),
        f"推荐模式真正 simultaneously-live peak 是 {recommended['simultaneously_live_peak_bytes']} bytes。",
        "是。checkpoint 为 locked O_APPEND canonical JSONL hash chain，绑定 remote identity、科学合同和 state/row transition SHA。",
        "不需要。只重做未完成对象；认证完成项跳过，ETag/size/LastModified/contract 改变或 orphan state 均 fail closed。",
        f"没有。未冻结的 voxel/range/normal/association 数值均标记 {PREPROCESSING_UNRESOLVED}。",
        f"没有。primary pair 仍为 {PRIMARY_MAP_SEQUENCE} -> {PRIMARY_QUERY_SEQUENCE}。",
        "没有。真实 lidar/*.bin 下载对象数和字节数均为 0。",
        "没有。Open3D/PCL/其他 registration 调用及 real result 均为 0。",
        (
        f"是；source-only 测试 {test_counts['collected']} collected, "
            f"{test_counts['passed']} passed, {test_counts['skipped']} skipped, 0 failed, 0 errors。"
        ),
        f"是；producer 返回及 freeze 前均强制运行 `{verifier_command}` 并要求 PASS，另有 11 个语义篡改用例必须全部被拒绝。",
        (
            "STAGE2_STORAGE_PLAN_READY="
            f"{str(bool(budget['CURRENT_DISK_SUFFICIENT'])).lower()}。"
        ),
        "下一步只能由用户在单独任务中授权真正 Stage-2 下载；本 closure 仍保持 STAGE2_DOWNLOAD_AUTHORIZED=false。",
    ]
    return [
        {"answer": answer, "question_number": index}
        for index, answer in enumerate(answers, start=1)
    ]


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Boreas v2 Stage-2 存储优化审计总结",
        "",
        f"- `STAGE2_STORAGE_PLAN_READY={str(summary['STAGE2_STORAGE_PLAN_READY']).lower()}`",
        f"- `CURRENT_DISK_SUFFICIENT={str(summary['CURRENT_DISK_SUFFICIENT']).lower()}`",
        "- `STAGE2_DOWNLOAD_AUTHORIZED=false`",
        "- `PUBLIC_DATA_V2_RUN_AUTHORIZED=false`",
        "- `REAL_REGISTRATION_AUTHORIZED=false`",
        "- `MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`",
        "- 本状态只表示存储与执行架构准备完成，不是下载或实验成功。",
        "",
        "## 27 个必答问题",
        "",
    ]
    for value in summary["answers"]:
        lines.append(f"{value['question_number']}. {value['answer']}")
        lines.append("")
    lines.extend(
        [
            "## 最终边界",
            "",
            summary["final_conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def _attestation_zeros() -> dict[str, Any]:
    return {
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "STAGE2_DOWNLOAD_AUTHORIZED": False,
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


def authenticate_pytest_junit(
    *, report_path: str | Path, expected_head: str, minimum_collected: int = 811
) -> dict[str, Any]:
    """Parse an immutable JUnit report instead of trusting CLI-supplied counts."""

    path = Path(report_path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise BoreasStage2StorageOptimizationError("pytest JUnit report is absent or unsafe")
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as error:
        raise BoreasStage2StorageOptimizationError("invalid pytest JUnit report") from error
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise BoreasStage2StorageOptimizationError("pytest JUnit report has no test suite")
    try:
        collected = sum(int(suite.attrib["tests"]) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    except (KeyError, ValueError) as error:
        raise BoreasStage2StorageOptimizationError("invalid pytest JUnit counts") from error
    testcase_count = sum(len(suite.findall("testcase")) for suite in suites)
    testcases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [
        (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        for case in testcases
    ]
    child_failures = sum(case.find("failure") is not None for case in testcases)
    child_errors = sum(case.find("error") is not None for case in testcases)
    child_skipped = sum(case.find("skipped") is not None for case in testcases)
    if testcase_count != collected:
        raise BoreasStage2StorageOptimizationError("pytest JUnit testcase count mismatch")
    if len(set(identities)) != testcase_count or any(not all(identity) for identity in identities):
        raise BoreasStage2StorageOptimizationError("pytest JUnit testcase identities invalid")
    if (child_failures, child_errors, child_skipped) != (failures, errors, skipped):
        raise BoreasStage2StorageOptimizationError("pytest JUnit child counters mismatch")
    passed = collected - failures - errors - skipped
    if (
        collected < minimum_collected
        or failures != 0
        or errors != 0
        or passed < 0
        or passed + skipped != collected
    ):
        raise BoreasStage2StorageOptimizationError("source-only pytest JUnit is not clean")
    return {
        "collected": collected,
        "errors": errors,
        "failed": failures,
        "head": expected_head,
        "junit_filename": "source_only_pytest_junit.xml",
        "junit_sha256": sha256_file(path),
        "junit_size_bytes": path.stat().st_size,
        "junit_verification_scope": (
            "INDEPENDENTLY_PARSED_SHA_BOUND_COUNTS_AND_UNIQUE_TESTCASE_IDENTITIES;"
            "THE_XML_IS_NOT_A_SIGNED_ATTESTATION"
        ),
        "minimum_collected_required": minimum_collected,
        "passed": passed,
        "skipped": skipped,
        "status": "PASS",
        "testcase_count": testcase_count,
    }


def build_boreas_stage2_storage_optimization(
    *,
    repository: str | Path,
    data_root: str | Path,
    runtime_root: str | Path,
    pytest_junit_xml: str | Path,
    require_clean_worktree: bool = True,
) -> dict[str, Any]:
    """Build a fresh, non-executing Stage-2 storage-plan runtime closure."""

    repository_path = Path(repository).resolve(strict=True)
    data_path = Path(data_root).resolve(strict=True)
    runtime_argument = Path(runtime_root)
    runtime_path = runtime_argument.resolve(strict=False)
    if runtime_argument != runtime_path or runtime_path.exists() or runtime_path.is_symlink():
        raise BoreasStage2StorageOptimizationError("fresh canonical absent runtime root required")
    environment = _assert_environment()
    repository_gate = inspect_repository_gate(
        repository_path, require_clean=require_clean_worktree
    )
    stage1 = inspect_frozen_stage1(repository_path)
    no_lidar_before = _assert_no_local_lidar_payload(data_path)
    stage1_root = repository_path / STAGE1_RELATIVE_ROOT
    planner_inputs = load_frozen_stage1_inputs(stage1_root)
    current_free_bytes = shutil.disk_usage(runtime_path.parent).free
    budget = optimized_storage_budgets(
        planner_inputs, current_free_bytes=current_free_bytes
    )
    if budget.get("CURRENT_DISK_SUFFICIENT") is not True:
        # A valid closure may still document false readiness, but this machine's
        # current requirement is to produce READY only when recommended mode fits.
        storage_ready = False
    else:
        storage_ready = True
    test_counts = authenticate_pytest_junit(
        report_path=pytest_junit_xml,
        expected_head=repository_gate["head"],
    )

    original = original_budget_breakdown(planner_inputs)
    if tuple(original[0]) != PLANNER_ORIGINAL_BREAKDOWN_FIELDS:
        raise BoreasStage2StorageOptimizationError("planner original breakdown schema changed")
    lifecycle = object_lifecycle_contract()
    if tuple(lifecycle[0]) != PLANNER_LIFECYCLE_FIELDS:
        raise BoreasStage2StorageOptimizationError("planner lifecycle schema changed")
    architecture = storage_architecture_contract(planner_inputs)
    science_locks = scientific_lock_contract(planner_inputs)
    no_lidar_guard: BoreasLidarPayloadGuard
    with NoRegistrationGuard() as no_registration_guard, BoreasLidarPayloadGuard() as no_lidar_guard:
        runtime_path.mkdir(parents=False, exist_ok=False)
        atomic_write_bytes(
            runtime_path / "source_only_pytest_junit.xml",
            Path(pytest_junit_xml).resolve(strict=True).read_bytes(),
        )
        atomic_write_json(runtime_path / "source_only_test_status.json", test_counts)
        old_budget = _load_json(stage1_root / "boreas_v2_stage2_disk_budget.json")
        atomic_write_csv(
            runtime_path / "stage2_storage_budget_original_breakdown.csv",
            original,
            PLANNER_ORIGINAL_BREAKDOWN_FIELDS,
        )
        atomic_write_json(
            runtime_path / "stage2_storage_budget_original_breakdown.json",
            {
                "breakdown_closes_to_required_safe_bytes": True,
                "legacy_formula": old_budget["formula"],
                "legacy_required_safe_disk_GiB": old_budget["required_safe_disk_GiB"],
                "legacy_required_safe_disk_bytes": old_budget["required_safe_disk_bytes"],
                "legacy_subtotal_before_safety_bytes": old_budget[
                    "subtotal_before_safety_bytes"
                ],
                "old_budget_sha256": STAGE1_OLD_BUDGET_SHA256,
                "rows": original,
                "schema_version": "boreas_v2_stage2_original_budget_breakdown_v1",
            },
        )
        atomic_write_bytes(
            runtime_path / "stage2_storage_budget_original_explanation.md",
            _original_explanation(old_budget, original).encode("utf-8"),
        )
        atomic_write_csv(
            runtime_path / "stage2_object_lifecycle_contract.csv",
            lifecycle,
            PLANNER_LIFECYCLE_FIELDS,
        )
        atomic_write_json(
            runtime_path / "stage2_object_lifecycle_contract.json",
            {
                "lifecycle_classes": [
                    "REMOTE_ONLY",
                    "TEMPORARY_DOWNLOAD",
                    "TEMPORARY_DECODED",
                    "PERSISTENT_RAW",
                    "PERSISTENT_CANONICAL",
                    "PERSISTENT_MAP",
                    "PERSISTENT_RESULT",
                    "CHECKPOINT",
                ],
                "rows": lifecycle,
                "schema_version": "boreas_v2_stage2_object_lifecycle_contract_v1",
                "two_pass_query_preferred": True,
            },
        )
        canonical = _canonical_bundle_contract()
        streaming_map = _streaming_map_contract(stage1)
        streaming_query = _streaming_query_contract(stage1)
        checkpoint = _checkpoint_contract()
        temp = _temp_contract()
        modes = _execution_modes_contract(budget)
        atomic_write_json(runtime_path / "canonical_bundle_storage_contract_v2.json", canonical)
        atomic_write_json(runtime_path / "stage2_streaming_map_contract.json", streaming_map)
        atomic_write_json(runtime_path / "stage2_streaming_query_contract.json", streaming_query)
        atomic_write_json(runtime_path / "stage2_checkpoint_contract.json", checkpoint)
        atomic_write_json(runtime_path / "stage2_temp_file_contract.json", temp)
        atomic_write_json(
            runtime_path / "boreas_v2_stage2_disk_budget_optimized.json", budget
        )
        atomic_write_csv(
            runtime_path / "boreas_v2_stage2_disk_budget_optimized.csv",
            _budget_csv_rows(budget),
            BUDGET_CSV_FIELDS,
        )
        atomic_write_json(runtime_path / "boreas_v2_stage2_execution_modes.json", modes)
        atomic_write_json(runtime_path / "stage2_scientific_lock_contract.json", science_locks)
        atomic_write_json(runtime_path / "stage2_storage_architecture_contract.json", architecture)

        no_lidar = {**no_lidar_guard.attestation(), **_attestation_zeros()}
        no_lidar.update(
            {
                "data_root": str(data_path),
                "initial_local_lidar_bin_count": len(no_lidar_before),
                "local_lidar_bin_count": 0,
                "metadata_only_audit": True,
                "payload_materialization_authorized": False,
                "status": "PASS",
            }
        )
        atomic_write_json(runtime_path / "NO_LIDAR_PAYLOAD_ATTESTATION.json", no_lidar)

        no_icp = no_registration_guard.attestation(runtime_path)
        no_icp.update(_attestation_zeros())
        no_icp.update(
            {
                "static_source_audit": assert_preparation_sources_are_safe(
                    repository_path / "src/phase_a_harness/real_data_preparation"
                ),
                "status": "PASS",
            }
        )
        if no_icp.get("pass") is not True:
            raise BoreasStage2StorageOptimizationError("NO-ICP attestation failed")
        atomic_write_json(runtime_path / "NO_ICP_ATTESTATION.json", no_icp)

        verifier_command = "python3 scripts/verify_boreas_v2_stage2_storage_plan.py"
        readiness = {
            **_attestation_zeros(),
            "CURRENT_DISK_SUFFICIENT": budget["CURRENT_DISK_SUFFICIENT"],
            "STAGE2_STORAGE_PLAN_READY": storage_ready,
            "architecture_test_scope": "SOURCE_ONLY_SYNTHETIC_FIXTURES",
            "audit_pre_change_source_only_baseline": {
                "collected": 811,
                "errors": 0,
                "failed": 0,
                "head": EXPECTED_TAG_COMMITS[STAGE1_TAG],
                "passed": 801,
                "skipped": 10,
                "status": "PASS",
            },
            "canonical_backend_input_no_duplication": True,
            "checkpoint_resume_contract_ready": True,
            "current_free_bytes": current_free_bytes,
            "independent_storage_verifier_required": True,
            "next_step_requires_separate_user_authorization": True,
            "planned_future_rich_snapshot_count": PLANNED_RICH_SNAPSHOT_COUNT,
            "planned_future_snapshot_count": PLANNED_SNAPSHOT_COUNT,
            "planned_future_backend_trial_count_after_success": 200,
            "planned_future_rich_interval_count": 10,
            "planned_future_snapshots_per_interval": 5,
            "planned_future_weak_interval_count": 10,
            "planned_future_weak_snapshot_count": PLANNED_WEAK_SNAPSHOT_COUNT,
            "preprocessing_parameter_status": PREPROCESSING_UNRESOLVED,
            "recommended_execution_mode": "STREAMING_LOW_DISK",
            "real_stage2_execution_runner_in_scope": False,
            "repository_gate": repository_gate,
            "runtime_disk_gate_contract_ready": True,
            "runtime_disk_gate_runner_integration_required": True,
            "runtime_disk_gate_status": (
                "PLANNER_START_AND_PROJECTED_WRITE_GUARDS_IMPLEMENTED_AND_SYNTHETICALLY_"
                "TESTED;SEPARATELY_AUTHORIZED_REAL_STAGE2_RUNNER_MUST_INVOKE_THEM"
            ),
            "schema_version": "boreas_v2_stage2_storage_readiness_v1",
            "source_only_test_status": test_counts,
            "stage1_binding": stage1,
            "stage1_frozen_assets_verified": True,
            "storage_planner_verifier_command": verifier_command,
            "target_physical_copy_count": 1,
        }
        atomic_write_json(
            runtime_path / "boreas_v2_stage2_storage_readiness.json", readiness
        )
        answers = _summary_answers(
            budget=budget, test_counts=test_counts, verifier_command=verifier_command
        )
        ready_text = str(storage_ready).lower()
        summary = {
            **readiness,
            "answers": answers,
            "answer_count": len(answers),
            "final_conclusion": (
                f"STAGE2_STORAGE_PLAN_READY={ready_text} 只表示 storage/lifecycle/resume/low-disk "
                "architecture 的当前审计结论；真实 Stage-2 runner 不在本任务范围，未来 runner "
                "必须在启动和每次 projected write 前接入已测试的磁盘门禁。所有真实下载、"
                "geometry selection、Open3D/PCL "
                "和 registration 仍未授权且未执行。下一步若要下载，必须由用户在单独任务中"
                "显式授权，并先独立冻结缺失的 preprocessing 科学参数。"
            ),
            "schema_version": "boreas_v2_stage2_storage_summary_v1",
        }
        atomic_write_json(runtime_path / "stage2_storage_summary.json", summary)
        atomic_write_bytes(
            runtime_path / "stage2_storage_summary.md",
            _summary_markdown(summary).encode("utf-8"),
        )

    if _assert_no_local_lidar_payload(data_path):
        raise BoreasStage2StorageOptimizationError("LiDAR payload appeared during audit")
    payload_names = sorted(
        path.name
        for path in runtime_path.iterdir()
        if path.is_file() and path.name not in {"frozen_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "actual_execution_count": 0,
        "allowlist_object_count": EXPECTED_ALLOWLIST_OBJECT_COUNT,
        "allowlist_remote_bytes": EXPECTED_REMOTE_BYTES,
        "allowlist_sha256": STAGE1_ALLOWLIST_SHA256,
        "architecture_only": True,
        "downloaded_lidar_payload_count": 0,
        "payload": [
            {
                "path": name,
                "sha256": sha256_file(runtime_path / name),
                "size_bytes": (runtime_path / name).stat().st_size,
            }
            for name in payload_names
        ],
        "producer_commit": repository_gate["head"],
        "registration_execution_count": 0,
        "schema_version": "boreas_v2_stage2_storage_optimization_manifest_v1",
        "stage1_frozen_manifest_file_sha256": STAGE1_FROZEN_MANIFEST_FILE_SHA256,
        "stage1_manifest_root_sha256": STAGE1_MANIFEST_ROOT_SHA256,
        "stage1_tag_commit": EXPECTED_TAG_COMMITS[STAGE1_TAG],
    }
    manifest["manifest_root_sha256"] = compact_sha256(manifest)
    atomic_write_json(runtime_path / "frozen_manifest.json", manifest)
    atomic_write_bytes(
        runtime_path / "SHA256SUMS",
        _sha256sums_payload(runtime_path, [*payload_names, "frozen_manifest.json"]),
    )
    # READY is not merely a producer assertion.  The independently implemented
    # verifier must accept the complete, immutable closure before this builder
    # is allowed to return success.  The verifier imports neither this module
    # nor the storage planner.
    from .boreas_stage2_storage_optimization_verifier import (
        verify_boreas_v2_stage2_storage_plan,
    )

    verification = verify_boreas_v2_stage2_storage_plan(
        repository=repository_path,
        runtime_root=runtime_path,
        data_root=data_path,
    )
    if (
        verification.get("STAGE2_STORAGE_PLAN_VERIFICATION_PASS") is not True
        or verification.get("STAGE2_STORAGE_PLAN_READY") is not storage_ready
    ):
        raise BoreasStage2StorageOptimizationError(
            "independent storage verifier did not accept the completed closure"
        )
    return summary


def freeze_boreas_stage2_storage_optimization(
    *,
    repository: str | Path,
    data_root: str | Path,
    runtime_root: str | Path,
    frozen_root: str | Path,
) -> None:
    from .boreas_stage2_storage_optimization_verifier import (
        verify_boreas_v2_stage2_storage_plan,
    )

    verification = verify_boreas_v2_stage2_storage_plan(
        repository=repository,
        runtime_root=runtime_root,
        data_root=data_root,
    )
    if verification.get("STAGE2_STORAGE_PLAN_VERIFICATION_PASS") is not True:
        raise BoreasStage2StorageOptimizationError(
            "independent storage verifier failed before freeze"
        )
    _freeze_runtime_assets(Path(runtime_root), Path(frozen_root))
    frozen_verification = verify_boreas_v2_stage2_storage_plan(
        repository=repository,
        runtime_root=frozen_root,
        data_root=data_root,
    )
    if frozen_verification.get("STAGE2_STORAGE_PLAN_VERIFICATION_PASS") is not True:
        raise BoreasStage2StorageOptimizationError(
            "independent storage verifier failed on copied frozen closure"
        )


__all__ = [
    "BoreasStage2StorageOptimizationError",
    "EXPECTED_ALLOWLIST_OBJECT_COUNT",
    "EXPECTED_REMOTE_BYTES",
    "PREPROCESSING_UNRESOLVED",
    "PRIMARY_MAP_SEQUENCE",
    "PRIMARY_QUERY_SEQUENCE",
    "build_boreas_stage2_storage_optimization",
    "freeze_boreas_stage2_storage_optimization",
    "inspect_frozen_stage1",
    "inspect_repository_gate",
]
