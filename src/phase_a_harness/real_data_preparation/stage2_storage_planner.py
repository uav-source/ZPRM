"""Deterministic, read-only storage accounting for Boreas v2 Stage-2.

This module plans *file lifetime and cache policy only*.  It deliberately has
no downloader, point-cloud decoder, registration backend, voxel size, or
geometry-selection parameter.  Point-cloud sizes below are upper bounds
derived from the frozen Stage-1 allowlist and Boreas' published six-float32
wire format; they are not claims about a future voxelized map's actual size.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import sha256_file


GIB = 1024**3
GB = 1_000_000_000

FROZEN_PRIMARY_MAP_SEQUENCE = "boreas-2021-11-14-09-47"
FROZEN_PRIMARY_QUERY_SEQUENCE = "boreas-2021-01-26-11-22"
FROZEN_ALLOWLIST_OBJECT_COUNT = 20_061
FROZEN_REMOTE_PAYLOAD_BYTES = 104_158_637_472
FROZEN_MAP_OBJECT_COUNT = 8_202
FROZEN_MAP_REMOTE_BYTES = 41_998_817_280
FROZEN_QUERY_OBJECT_COUNT = 11_859
FROZEN_QUERY_REMOTE_BYTES = 62_159_820_192
FROZEN_SNAPSHOT_COUNT = 100
FROZEN_WEAK_SNAPSHOT_COUNT = 50
FROZEN_RICH_SNAPSHOT_COUNT = 50
FROZEN_WEAK_INTERVAL_COUNT = 10
FROZEN_RICH_INTERVAL_COUNT = 10
FROZEN_SNAPSHOTS_PER_INTERVAL = 5
FROZEN_BACKEND_TRIAL_COUNT = 200

FROZEN_STAGE1_MANIFEST_SHA256 = (
    "71cfd78aa588c67dd28f8f8be87b7514c20ed090e75a8433583362253a97a8be"
)
FROZEN_STAGE1_SHA256SUMS_SHA256 = (
    "cbda458050de5bd3571d18503f7e7db54277e22e299b61116f5a9801df049967"
)
FROZEN_PAIR_SELECTION_SHA256 = (
    "b28a52498a2fddc1b80d43c626b75179358066677ce8f493be22afba7a19639c"
)
FROZEN_ALLOWLIST_SHA256 = (
    "26ac211c854472dcb3db2f1cd5b096849bfd27867bac34e75bc8ce0d01bb2787"
)
FROZEN_DOWNLOAD_PLAN_SHA256 = (
    "3687a4e53945a97fd88e7f32ee2f57f5a0ddb0cac1e24ad452d047df218caa0b"
)
FROZEN_DISK_BUDGET_SHA256 = (
    "f8179dbc672a3ac08ada50f2835f86fec5e4a6b651d6d5cca6a61fc4fbd991a5"
)

PREPROCESSING_PARAMETER_STATUS = "PREPROCESSING_PARAMETER_REQUIRES_STAGE2_PREREGISTRATION"
PLANNER_CONFIGURABLE_FIELDS = frozenset(
    {
        "execution_mode",
        "current_free_bytes",
        "temporary_directory",
        "persistent_directory",
    }
)
SCIENTIFIC_LOCKED_FIELDS = frozenset(
    {
        "primary_pair",
        "reserve_pairs",
        "allowlist_object_count",
        "remote_payload_bytes",
        "voxel_size",
        "range_limits",
        "normal_neighborhood",
        "association_distance",
        "geometry_metric",
        "weak_rich_thresholds",
        "snapshot_count",
        "weak_snapshot_count",
        "rich_snapshot_count",
        "weak_interval_count",
        "rich_interval_count",
        "snapshots_per_interval",
        "backend_trial_count",
        "icp_parameters",
    }
)


class Stage2StoragePlanError(RuntimeError):
    """A frozen input, accounting identity, or storage gate failed."""


class ScientificParameterModificationForbidden(Stage2StoragePlanError):
    """A storage-only planner was asked to change scientific semantics."""


class InsufficientStage2DiskError(Stage2StoragePlanError):
    """Available disk is below a fail-closed mode threshold."""


@dataclass(frozen=True)
class Stage1StorageInputs:
    stage1_root: Path
    map_sequence_id: str
    query_sequence_id: str
    allowlist_object_count: int
    remote_payload_bytes: int
    map_object_count: int
    map_remote_bytes: int
    query_object_count: int
    query_remote_bytes: int
    maximum_map_object_bytes: int
    maximum_query_object_bytes: int
    allowlist_csv_bytes: int
    frozen_closure_bytes: int
    old_required_safe_bytes: int
    old_subtotal_bytes: int

    @property
    def maximum_object_bytes(self) -> int:
        return max(self.maximum_map_object_bytes, self.maximum_query_object_bytes)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise Stage2StoragePlanError(f"frozen Stage-1 file changed: {path.name}")


def load_frozen_stage1_inputs(stage1_root: str | Path) -> Stage1StorageInputs:
    """Authenticate and summarize the frozen Stage-1 storage inputs."""

    root = Path(stage1_root).resolve(strict=True)
    expected_hashes = {
        "frozen_manifest.json": FROZEN_STAGE1_MANIFEST_SHA256,
        "SHA256SUMS": FROZEN_STAGE1_SHA256SUMS_SHA256,
        "boreas_v2_pair_selection.json": FROZEN_PAIR_SELECTION_SHA256,
        "boreas_v2_stage2_download_allowlist.csv": FROZEN_ALLOWLIST_SHA256,
        "boreas_v2_stage2_download_plan.json": FROZEN_DOWNLOAD_PLAN_SHA256,
        "boreas_v2_stage2_disk_budget.json": FROZEN_DISK_BUDGET_SHA256,
    }
    for name, expected in expected_hashes.items():
        _require_sha(root / name, expected)
    pair = _load_json(root / "boreas_v2_pair_selection.json")["PRIMARY_PAIR"]
    plan = _load_json(root / "boreas_v2_stage2_download_plan.json")
    old = _load_json(root / "boreas_v2_stage2_disk_budget.json")
    map_id = str(pair["map_sequence_id"])
    query_id = str(pair["query_sequence_id"])
    if (map_id, query_id) != (
        FROZEN_PRIMARY_MAP_SEQUENCE,
        FROZEN_PRIMARY_QUERY_SEQUENCE,
    ):
        raise Stage2StoragePlanError("frozen primary pair changed")

    counts = {"TARGET_MAP": 0, "QUERY": 0}
    byte_totals = {"TARGET_MAP": 0, "QUERY": 0}
    maxima = {"TARGET_MAP": 0, "QUERY": 0}
    allowlist = root / "boreas_v2_stage2_download_allowlist.csv"
    with allowlist.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "selection_role",
            "sequence_id",
            "key",
            "timestamp_us",
            "last_modified",
            "size_bytes",
            "selection_reason",
        }
        if set(reader.fieldnames or ()) != required:
            raise Stage2StoragePlanError("allowlist schema changed")
        for row in reader:
            role = row["selection_role"]
            if role not in counts:
                raise Stage2StoragePlanError(f"unknown allowlist role: {role}")
            expected_sequence = map_id if role == "TARGET_MAP" else query_id
            if row["sequence_id"] != expected_sequence:
                raise Stage2StoragePlanError("allowlist sequence/pair binding changed")
            size = int(row["size_bytes"])
            if size <= 0:
                raise Stage2StoragePlanError("allowlist contains nonpositive object size")
            counts[role] += 1
            byte_totals[role] += size
            maxima[role] = max(maxima[role], size)

    if (
        counts != {"TARGET_MAP": FROZEN_MAP_OBJECT_COUNT, "QUERY": FROZEN_QUERY_OBJECT_COUNT}
        or byte_totals
        != {"TARGET_MAP": FROZEN_MAP_REMOTE_BYTES, "QUERY": FROZEN_QUERY_REMOTE_BYTES}
        or sum(counts.values()) != FROZEN_ALLOWLIST_OBJECT_COUNT
        or sum(byte_totals.values()) != FROZEN_REMOTE_PAYLOAD_BYTES
    ):
        raise Stage2StoragePlanError("frozen allowlist counts or bytes changed")
    selected = plan["selected_objects"]
    if (
        int(plan["estimated_download_bytes"]) != FROZEN_REMOTE_PAYLOAD_BYTES
        or int(selected["TARGET_MAP"]["object_count"]) != counts["TARGET_MAP"]
        or int(selected["TARGET_MAP"]["remote_bytes"]) != byte_totals["TARGET_MAP"]
        or int(selected["QUERY"]["object_count"]) != counts["QUERY"]
        or int(selected["QUERY"]["remote_bytes"]) != byte_totals["QUERY"]
    ):
        raise Stage2StoragePlanError("Stage-1 download plan disagrees with its allowlist")
    closure_names = expected_hashes.keys()
    closure_bytes = sum((root / name).stat().st_size for name in closure_names)
    return Stage1StorageInputs(
        stage1_root=root,
        map_sequence_id=map_id,
        query_sequence_id=query_id,
        allowlist_object_count=sum(counts.values()),
        remote_payload_bytes=sum(byte_totals.values()),
        map_object_count=counts["TARGET_MAP"],
        map_remote_bytes=byte_totals["TARGET_MAP"],
        query_object_count=counts["QUERY"],
        query_remote_bytes=byte_totals["QUERY"],
        maximum_map_object_bytes=maxima["TARGET_MAP"],
        maximum_query_object_bytes=maxima["QUERY"],
        allowlist_csv_bytes=allowlist.stat().st_size,
        frozen_closure_bytes=closure_bytes,
        old_required_safe_bytes=int(old["required_safe_disk_bytes"]),
        old_subtotal_bytes=int(old["subtotal_before_safety_bytes"]),
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


def original_budget_breakdown(inputs: Stage1StorageInputs) -> list[dict[str, Any]]:
    """Explain the old 990.39 GiB envelope without inventing subcomponents.

    The frozen v1 budget has only five aggregate byte terms.  Its internal
    subcomponents were never measured, so those rows remain explicit
    ``NOT_SEPARATELY_QUANTIFIED`` zeros instead of allocating invented bytes.
    """

    raw = inputs.remote_payload_bytes
    map_raw = inputs.map_remote_bytes

    def row(
        component: str,
        raw_bytes: int,
        factor: float,
        lifetime: str,
        retained: bool,
        reason: str,
        source: str,
        *,
        simultaneous: bool = True,
        units: int = 1,
        bytes_per_unit: int | str | None = None,
    ) -> dict[str, Any]:
        peak = int(raw_bytes * factor) if simultaneous else 0
        return {
            "component": component,
            "unit_count": units,
            "bytes_per_unit": raw_bytes if bytes_per_unit is None else bytes_per_unit,
            "raw_bytes": raw_bytes,
            "duplication_factor": factor,
            "lifetime": lifetime,
            "simultaneously_live": simultaneous,
            "peak_contribution_bytes": peak,
            "retained_after_stage2": retained,
            "reason": reason,
            "source_of_estimate": source,
        }

    rows = [
        row(
            "raw lidar payload",
            raw,
            1.0,
            "PERSISTENT_RAW",
            True,
            "Old envelope assumed the full selected raw payload is local.",
            "frozen disk budget: download_bytes",
            units=inputs.allowlist_object_count,
            bytes_per_unit="PER_OBJECT_ALLOWLIST_SIZE",
        ),
        row(
            "temporary download files",
            raw,
            1.0,
            "TEMPORARY_DOWNLOAD",
            False,
            "Old envelope reserved another complete-payload temporary copy.",
            "frozen disk budget: temporary_processing_bytes",
        ),
        row(
            "decoded arrays",
            raw,
            2.0,
            "TEMPORARY_DECODED",
            False,
            "Boreas six-float32 wire rows become six-float64 devkit arrays.",
            "frozen disk budget: decoded_or_unpacked_working_bytes",
        ),
        row(
            "map scan materialization",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Not itemized inside the frozen aggregate decoded-working term.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "accumulated target map",
            map_raw,
            2.0,
            "PERSISTENT_MAP",
            True,
            "Old envelope doubled the selected map-side raw bytes.",
            "frozen disk budget: target_map_bytes",
            units=1,
        ),
        row(
            "voxel map copies",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "No voxel size or measured voxel-map size was frozen in Stage-1.",
            PREPROCESSING_PARAMETER_STATUS,
        ),
        row(
            "query decoded points",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Included but not itemized in the two-times decoded aggregate.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "canonical source arrays",
            raw,
            2.0,
            "PERSISTENT_CANONICAL",
            True,
            "Old envelope used two payload-equivalents for the canonical bundle.",
            "frozen disk budget: canonical_bundle_bytes",
        ),
        row(
            "canonical target arrays",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Not split from the aggregate canonical-bundle term.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "Open3D copies",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Backend-specific copies were not separately measured.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "PCL copies",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Backend-specific copies were not separately measured.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "per-snapshot target copies",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "The old aggregate did not assert or quantify 100 physical maps.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "intermediate PCD/bin files",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "Potential temporaries were not itemized in Stage-1.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "geometry metrics",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            True,
            "Small results were not separately estimated.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "publication artifacts",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            True,
            "Publication outputs were not separately estimated.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "resume/checkpoint files",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            True,
            "Resume state was not separately estimated.",
            "not independently specified by frozen v1 budget",
        ),
        row(
            "safety factor",
            inputs.old_subtotal_bytes,
            0.5,
            "SAFETY_MARGIN",
            False,
            "The frozen envelope added 50 percent to the naive subtotal.",
            "frozen disk budget: minimum_safety_multiplier=1.5",
        ),
        row(
            "filesystem overhead",
            0,
            0.0,
            "NOT_SEPARATELY_QUANTIFIED",
            False,
            "No independent filesystem-overhead term was specified.",
            "not independently specified by frozen v1 budget",
        ),
    ]
    subtotal = sum(
        item["peak_contribution_bytes"]
        for item in rows
        if item["component"] != "safety factor"
    )
    total = sum(item["peak_contribution_bytes"] for item in rows)
    if subtotal != inputs.old_subtotal_bytes or total != inputs.old_required_safe_bytes:
        raise Stage2StoragePlanError("original budget breakdown does not close arithmetically")
    return rows


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


def object_lifecycle_contract() -> list[dict[str, Any]]:
    """Return the normative low-disk object lifecycle rows."""

    rows = [
        (
            "map remote object", "REMOTE_ONLY", "frozen S3 key", "map streaming pass",
            "never local by default", False, False, "identity rechecked",
        ),
        (
            "map raw temp", "TEMPORARY_DOWNLOAD", "one authenticated download", "decoder",
            "delete immediately after state transition", False, False, "partial file rejected",
        ),
        (
            "map decoded scan", "TEMPORARY_DECODED",
            "six-float32 raw decoded to float64", "GT transform and accumulator",
            "delete after accumulator commit", False, False, "rederive from object",
        ),
        (
            "raw cache object", "PERSISTENT_RAW", "optional FULL_RAW_CACHE", "decoder",
            "retain only in full-cache mode", True, False, "identity must remain frozen",
        ),
        (
            "selected canonical source", "PERSISTENT_CANONICAL",
            "second-pass selected query", "both backends", "content-addressed retention",
            True, True, "verify SHA; never silently replace",
        ),
        (
            "content-addressed target map", "PERSISTENT_MAP",
            "incremental deterministic builder", "100 snapshot references and both backends",
            "one physical copy", True, True, "checkpoint or deterministic rebuild",
        ),
        (
            "geometry/result row", "PERSISTENT_RESULT", "query screen or backend output",
            "selection/analysis/verifier", "retain immutable row and SHA", True, True,
            "row SHA must match",
        ),
        (
            "map checkpoint", "CHECKPOINT", "atomic accumulator checkpoint", "strict resume",
            "retain latest authenticated generation", True, True, "orphan generation rejected",
        ),
        (
            "query raw temp", "TEMPORARY_DOWNLOAD", "one authenticated download",
            "first-pass metric or second-pass source", "delete immediately after consumption",
            False, False, "partial file rejected",
        ),
        (
            "query decoded scan", "TEMPORARY_DECODED", "raw decoder",
            "geometry metric or canonical writer", "delete immediately after consumption",
            False, False, "rederive from object",
        ),
        (
            "PCL source/target PCD", "TEMPORARY_DECODED",
            "deterministic conversion from canonical NPY", "PCL CLI only",
            "delete when trial exits", False, False, "rederive; canonical SHA is authority",
        ),
    ]
    return [dict(zip(LIFECYCLE_FIELDS, row)) for row in rows]


def scientific_lock_contract(inputs: Stage1StorageInputs) -> dict[str, Any]:
    return {
        "planner_configurable_fields": sorted(PLANNER_CONFIGURABLE_FIELDS),
        "storage_only_changes": [
            "data_lifecycle",
            "file_organization",
            "cache_policy",
            "content_addressing",
            "temporary_file_cleanup",
            "streaming_architecture",
        ],
        "scientific_fields_planner_configurable": False,
        "scientific_locked_fields": sorted(SCIENTIFIC_LOCKED_FIELDS),
        "primary_pair": {
            "map_sequence_id": inputs.map_sequence_id,
            "query_sequence_id": inputs.query_sequence_id,
        },
        "allowlist_object_count": inputs.allowlist_object_count,
        "remote_payload_bytes": inputs.remote_payload_bytes,
        "future_success_design": {
            "weak_interval_count": FROZEN_WEAK_INTERVAL_COUNT,
            "rich_interval_count": FROZEN_RICH_INTERVAL_COUNT,
            "snapshots_per_interval": FROZEN_SNAPSHOTS_PER_INTERVAL,
            "weak_snapshot_count": FROZEN_WEAK_SNAPSHOT_COUNT,
            "rich_snapshot_count": FROZEN_RICH_SNAPSHOT_COUNT,
            "snapshot_count": FROZEN_SNAPSHOT_COUNT,
            "backend_trial_count": FROZEN_BACKEND_TRIAL_COUNT,
        },
        "preprocessing_parameter_status": PREPROCESSING_PARAMETER_STATUS,
        "voxel_size": PREPROCESSING_PARAMETER_STATUS,
    }


def assert_scientific_locks(candidate: Mapping[str, Any]) -> None:
    """Reject every scientific parameter presented to the storage planner."""

    def nested_keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                found.add(str(key))
                found.update(nested_keys(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.update(nested_keys(child))
        return found

    forbidden = sorted(nested_keys(candidate).intersection(SCIENTIFIC_LOCKED_FIELDS))
    unknown = sorted(set(candidate).difference(PLANNER_CONFIGURABLE_FIELDS))
    if forbidden or unknown:
        names = sorted(set(forbidden + unknown))
        raise ScientificParameterModificationForbidden(
            f"storage planner cannot configure scientific fields: {names}"
        )
    if "execution_mode" in candidate and candidate["execution_mode"] not in {
        "STREAMING_LOW_DISK",
        "FULL_RAW_CACHE",
    }:
        raise Stage2StoragePlanError("invalid execution_mode")
    for field in ("temporary_directory", "persistent_directory"):
        if field in candidate and not isinstance(candidate[field], (str, Path)):
            raise Stage2StoragePlanError(f"{field} must be a path-like string")
    if "current_free_bytes" in candidate and (
        isinstance(candidate["current_free_bytes"], bool)
        or not isinstance(candidate["current_free_bytes"], int)
        or candidate["current_free_bytes"] < 0
    ):
        raise Stage2StoragePlanError("current_free_bytes must be a nonnegative integer")


def _derived_size_estimates(inputs: Stage1StorageInputs) -> dict[str, int]:
    # Boreas lidar raw: N x 6 little-endian float32 (24 bytes/point).
    # Persistent source/target: N x 3 float64 (also 24 bytes/point), so raw
    # bytes are a conservative exact-format upper bound before any future,
    # separately preregistered preprocessing.
    target_upper = inputs.map_remote_bytes
    selected_source_upper = FROZEN_SNAPSHOT_COUNT * (
        inputs.maximum_query_object_bytes + 128  # NPY v1.0 header upper bound.
    )
    # Durable receipts/checkpoints/results are not yet materialized, so quote a
    # transparent metadata upper bound: one frozen allowlist row-equivalent for
    # every retained evidence family, plus the authenticated Stage-1 closure.
    metadata_family_count = 19
    evidence_upper = (
        inputs.frozen_closure_bytes + metadata_family_count * inputs.allowlist_csv_bytes
    )
    # PCL writes float32 XYZ, half of the six-float32 raw input size, with a
    # bounded header. It converts the one canonical target and one source only.
    pcl_temp_upper = (
        target_upper // 2
        + 4096
        + inputs.maximum_query_object_bytes // 2
        + 4096
        + 1 * 1024**2
    )
    one_scan_streaming_upper = (
        inputs.maximum_object_bytes  # temporary raw
        + 2 * inputs.maximum_object_bytes  # six-float64 decoded
        + inputs.maximum_object_bytes  # conservative transformed working array
        + 1 * 1024**2
    )
    map_checkpoint_upper = target_upper + 128
    persistent_without_raw = target_upper + 128 + selected_source_upper + evidence_upper
    return {
        "target_map_upper_bound_bytes": target_upper + 128,
        "selected_sources_upper_bound_bytes": selected_source_upper,
        "small_evidence_upper_bound_bytes": evidence_upper,
        "pcl_runtime_conversion_upper_bound_bytes": pcl_temp_upper,
        "single_scan_streaming_upper_bound_bytes": one_scan_streaming_upper,
        "map_checkpoint_upper_bound_bytes": map_checkpoint_upper,
        "persistent_without_raw_bytes": persistent_without_raw,
    }


def _mode_budget(
    *, mode: str, persistent_bytes: int, temporary_peak_bytes: int,
    safety_rate: float, current_free_bytes: int,
) -> dict[str, Any]:
    live = persistent_bytes + temporary_peak_bytes
    margin = math.ceil(live * safety_rate)
    recommended = live + margin
    return {
        "mode": mode,
        "persistent_bytes": persistent_bytes,
        "temporary_peak_bytes": temporary_peak_bytes,
        "simultaneously_live_peak_bytes": live,
        "safety_margin_fraction": safety_rate,
        "safety_margin_bytes": margin,
        "recommended_free_disk_bytes": recommended,
        "persistent_GiB": persistent_bytes / GIB,
        "temporary_peak_GiB": temporary_peak_bytes / GIB,
        "simultaneously_live_peak_GiB": live / GIB,
        "safety_margin_GiB": margin / GIB,
        "recommended_free_disk_GiB": recommended / GIB,
        "current_free_bytes": int(current_free_bytes),
        "CURRENT_DISK_SUFFICIENT": current_free_bytes >= recommended,
        "additional_bytes_required": max(0, recommended - current_free_bytes),
        "additional_GiB_required": max(0, recommended - current_free_bytes) / GIB,
        "minimum_free_disk_required_before_start_bytes": recommended,
        # The full live-peak envelope is a pre-start admission threshold.  At
        # runtime, aborting at that same value would stop after the first
        # writes and make the planned peak unreachable.  The runtime floor is
        # the reserved safety margin; callers additionally account for the
        # projected next atomic write (see ``assert_runtime_watermark``).
        "abort_if_free_disk_below_bytes": margin,
        "runtime_low_disk_watermark_bytes": margin,
        "watermark_definition": (
            "remaining free bytes after the projected next atomic write must be at least "
            "the mode safety margin"
        ),
        "watermark_action": "FAIL_CLOSED_BEFORE_NEXT_WRITE",
    }


def optimized_storage_budgets(
    inputs: Stage1StorageInputs, *, current_free_bytes: int
) -> dict[str, Any]:
    """Compute three lifecycle-aware peak budgets from frozen object sizes."""

    if isinstance(current_free_bytes, bool) or int(current_free_bytes) < 0:
        raise ValueError("current_free_bytes must be a nonnegative integer")
    current = int(current_free_bytes)
    estimates = _derived_size_estimates(inputs)
    persistent = estimates["persistent_without_raw_bytes"]
    theoretical_temp = estimates["pcl_runtime_conversion_upper_bound_bytes"]
    operational_temp = max(
        estimates["map_checkpoint_upper_bound_bytes"],
        estimates["pcl_runtime_conversion_upper_bound_bytes"],
        estimates["single_scan_streaming_upper_bound_bytes"],
    )
    modes = [
        _mode_budget(
            mode="THEORETICAL_MINIMUM",
            persistent_bytes=persistent,
            temporary_peak_bytes=theoretical_temp,
            safety_rate=0.10,
            current_free_bytes=current,
        ),
        _mode_budget(
            mode="RECOMMENDED_OPERATIONAL",
            persistent_bytes=persistent,
            temporary_peak_bytes=operational_temp,
            safety_rate=0.20,
            current_free_bytes=current,
        ),
        _mode_budget(
            mode="CONSERVATIVE_FULL_CACHE",
            persistent_bytes=persistent + inputs.remote_payload_bytes,
            temporary_peak_bytes=operational_temp,
            safety_rate=0.20,
            current_free_bytes=current,
        ),
    ]
    by_mode = {row["mode"]: row for row in modes}
    recommended = by_mode["RECOMMENDED_OPERATIONAL"]
    return {
        "schema_version": "boreas_v2_stage2_storage_budget_optimized_v1",
        "accounting_rule": (
            "persistent footprint plus the maximum simultaneously-live temporary phase; "
            "mutually exclusive temporary phases are never summed"
        ),
        "size_estimates": estimates,
        "estimate_limitations": {
            "actual_target_map_size": "UNKNOWN_UNTIL_STAGE2_PREPROCESSING_PREREGISTRATION",
            "target_map_budget_method": (
                "unvoxelized XYZ float64 upper bound derived from map-side raw bytes; "
                "no voxel size is assumed; this bound requires the separately preregistered "
                "transform/canonicalization contract to preserve or reduce point count"
            ),
            "selected_source_budget_method": (
                "100 times the largest query object projected to float64 XYZ plus an NPY "
                "header; requires the separately preregistered canonicalizer to preserve or "
                "reduce point count"
            ),
            "metadata_budget_method": (
                "Stage-1 closure bytes plus 19 allowlist-CSV-equivalent evidence families"
            ),
            "map_checkpoint_budget_method": (
                "one preallocated float64 XYZ replay array whose authenticated fixed-order "
                "scan ranges rebuild the accumulator locally after a crash; the synthetic "
                "JSON checkpoint is fixture-only and is not used for this production bound"
            ),
            "preprocessing_parameter_status": PREPROCESSING_PARAMETER_STATUS,
        },
        "modes": modes,
        "CURRENT_DISK_SUFFICIENT": recommended["CURRENT_DISK_SUFFICIENT"],
        "minimum_additional_GiB_required": by_mode["THEORETICAL_MINIMUM"][
            "additional_GiB_required"
        ],
        "recommended_additional_GiB_required": recommended["additional_GiB_required"],
        "minimum_free_disk_required_before_start_bytes": recommended[
            "recommended_free_disk_bytes"
        ],
        "abort_if_free_disk_below_bytes": recommended[
            "abort_if_free_disk_below_bytes"
        ],
        "runtime_low_disk_watermark_bytes": recommended[
            "runtime_low_disk_watermark_bytes"
        ],
        "STAGE2_DOWNLOAD_AUTHORIZED": False,
        "PUBLIC_DATA_V2_RUN_AUTHORIZED": False,
        "REAL_REGISTRATION_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "downloaded_lidar_payload_count": 0,
        "downloaded_lidar_bytes": 0,
        "registration_execution_count": 0,
    }


def assert_free_disk(
    current_free_bytes: int,
    budget: Mapping[str, Any],
    *,
    mode: str = "RECOMMENDED_OPERATIONAL",
) -> None:
    rows = {str(row["mode"]): row for row in budget["modes"]}
    if mode not in rows:
        raise ValueError(f"unknown storage mode: {mode}")
    required = int(rows[mode]["recommended_free_disk_bytes"])
    if current_free_bytes < required:
        raise InsufficientStage2DiskError(
            f"{mode} requires {required} free bytes; only {current_free_bytes} available"
        )


def assert_runtime_watermark(
    current_free_bytes: int,
    projected_next_write_bytes: int,
    budget: Mapping[str, Any],
    *,
    mode: str = "RECOMMENDED_OPERATIONAL",
) -> None:
    """Fail before a write that would consume the mode's safety reserve."""

    if (
        isinstance(current_free_bytes, bool)
        or isinstance(projected_next_write_bytes, bool)
        or int(current_free_bytes) < 0
        or int(projected_next_write_bytes) < 0
    ):
        raise ValueError("disk and projected-write byte counts must be nonnegative integers")
    rows = {str(row["mode"]): row for row in budget["modes"]}
    if mode not in rows:
        raise ValueError(f"unknown storage mode: {mode}")
    watermark = int(rows[mode]["runtime_low_disk_watermark_bytes"])
    remaining = int(current_free_bytes) - int(projected_next_write_bytes)
    if remaining < watermark:
        raise InsufficientStage2DiskError(
            f"{mode} projected remaining free bytes {remaining} are below runtime "
            f"watermark {watermark}"
        )


def storage_architecture_contract(inputs: Stage1StorageInputs) -> dict[str, Any]:
    """Return the scientific-equivalence and single-copy invariants."""

    return {
        "unique_target_map_count": 1,
        "snapshot_target_reference_count": FROZEN_SNAPSHOT_COUNT,
        "physical_target_map_copy_count": 1,
        "per_snapshot_target_copy_allowed": False,
        "target_store": "target_maps/<target_map_sha256>/target_points.npy",
        "snapshot_target_binding": "target_map_sha256",
        "canonical_source_physical_copy_count_per_snapshot": 1,
        "canonical_target_physical_copy_count": 1,
        "future_open3d_source_sha256_equals_future_pcl_source_sha256": True,
        "future_open3d_target_sha256_equals_future_pcl_target_sha256": True,
        "PCL_conversion": "DETERMINISTIC_TEMPORARY_DERIVATION_FROM_CANONICAL_NPY",
        "PCL_temporary_input_retained": False,
        "execution_modes": {
            "STREAMING_LOW_DISK": {
                "persistent_raw_payload": False,
                "two_pass_query_selection": True,
            },
            "FULL_RAW_CACHE": {
                "persistent_raw_payload": True,
                "two_pass_query_selection": True,
            },
        },
        "mode_scientific_equivalence_required": [
            "target_map_sha256",
            "geometry_metrics_sha256",
            "selection_manifest_sha256",
            "selected_canonical_source_sha256",
            "backend_results",
        ],
        "stage1_primary_pair": {
            "map_sequence_id": inputs.map_sequence_id,
            "query_sequence_id": inputs.query_sequence_id,
        },
        "preprocessing_parameter_status": PREPROCESSING_PARAMETER_STATUS,
    }


__all__ = [
    "FROZEN_ALLOWLIST_OBJECT_COUNT",
    "FROZEN_BACKEND_TRIAL_COUNT",
    "FROZEN_MAP_OBJECT_COUNT",
    "FROZEN_MAP_REMOTE_BYTES",
    "FROZEN_PRIMARY_MAP_SEQUENCE",
    "FROZEN_PRIMARY_QUERY_SEQUENCE",
    "FROZEN_QUERY_OBJECT_COUNT",
    "FROZEN_QUERY_REMOTE_BYTES",
    "FROZEN_REMOTE_PAYLOAD_BYTES",
    "FROZEN_SNAPSHOT_COUNT",
    "GIB",
    "InsufficientStage2DiskError",
    "LIFECYCLE_FIELDS",
    "ORIGINAL_BREAKDOWN_FIELDS",
    "PREPROCESSING_PARAMETER_STATUS",
    "SCIENTIFIC_LOCKED_FIELDS",
    "ScientificParameterModificationForbidden",
    "Stage1StorageInputs",
    "Stage2StoragePlanError",
    "assert_free_disk",
    "assert_runtime_watermark",
    "assert_scientific_locks",
    "load_frozen_stage1_inputs",
    "object_lifecycle_contract",
    "optimized_storage_budgets",
    "original_budget_breakdown",
    "scientific_lock_contract",
    "storage_architecture_contract",
]
