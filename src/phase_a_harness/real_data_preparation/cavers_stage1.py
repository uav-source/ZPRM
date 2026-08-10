"""CAVERS metadata/GT/calibration-only Stage-1 eligibility producer.

Only HTTP byte ranges for ZIP indexes and explicitly allow-listed small members
are materialized.  No point-cloud payload or registration implementation is
reachable from this module.
"""

from __future__ import annotations

import binascii
import csv
import hashlib
import http.client
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    compact_sha256,
    sha256_file,
)
from .stage1_gt_overlap import (
    FROZEN_STAGE1_OVERLAP_CONTRACT,
    compute_stage1_gt_overlap,
    rank_distinct_stage1_pairs,
)
from .transforms import compose_world_sensor, transform_from_xyzw


ZENODO_RECORD_ID = 19367714
ZENODO_CONCEPT_RECORD_ID = "19367713"
ZENODO_DOI = "10.5281/zenodo.19367714"
ZENODO_VERSION = "0.0.1"
ZENODO_REVISION = 18
ZENODO_LICENSE = "cc-by-4.0"
ZENODO_PUBLICATION_DATE = "2026-04-01"
ZENODO_TOTAL_ARCHIVE_BYTES = 162_022_890_726
ZENODO_METADATA_SHA256 = "5047d09c5c7e39f0df861b1773edf1304aa1d0448bbc3a82b7396aae147f0c24"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
GITHUB_URL = "https://github.com/spaceuma/cavers.git"
GITHUB_COMMIT = "74ead2bf1cffa337a1bff1b03b990819cb7ff067"
GITHUB_CODE_LICENSE = "MIT"
ARXIV_ID = "2604.15052"
ARXIV_PDF_SHA256 = "ea7cfc4385c768b3144958952d0923ccd09e88f4b56c3814544328b698c98029"
EXPECTED_BRANCH = "prep/cavers-single-dataset-stage1-v1"
MAX_STAGE1_FILE_BYTES = 500_000_000
TF_EXPORTED_QUATERNION_MAX_NORM_ERROR = 1e-6
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"

CANDIDATE_SEQUENCES = tuple(
    [f"loc_diablo_{index}" for index in range(1, 9)]
    + [f"loc_handheld_{index}" for index in range(1, 5)]
)
EXCLUDED_NO_GT_SEQUENCES = ("loc_handheld_5", "loc_handheld_6")
RAW_MEMBER_SUFFIXES = (
    "GT_ODOM/data.csv",
    "TF/data.csv",
    "TF_STATIC/data.csv",
    "VELODYNE_CLOUD/data.csv",
)
ROSBAG_MEMBER_SUFFIX = "metadata.yaml"
ALLOWED_UNCERTAINTY_TYPES = {
    "1SIGMA",
    "RMSE",
    "MANUFACTURER_ACCURACY",
    "CONFIDENCE_BOUND",
    "CONSERVATIVE_BOUND",
    "UNKNOWN",
}
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


class CaversStage1Error(RuntimeError):
    """Fail-closed CAVERS preparation error."""


class Stage1LargeFileDownloadForbidden(CaversStage1Error):
    """A request attempted to materialize an oversized file."""


RESUME_POLICY = (
    "authenticate completed sequence checkpoints by official archive identity, "
    "size, CRC32, and SHA-256 before continuation"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_text(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.check_output(list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT)


def assert_full_download_allowed(size_bytes: int, maximum_bytes: int) -> None:
    if maximum_bytes <= 0 or maximum_bytes > MAX_STAGE1_FILE_BYTES:
        raise Stage1LargeFileDownloadForbidden(
            f"max-single-download-bytes must be in [1,{MAX_STAGE1_FILE_BYTES}]"
        )
    if size_bytes > maximum_bytes:
        raise Stage1LargeFileDownloadForbidden(
            f"STAGE1_LARGE_FILE_DOWNLOAD_FORBIDDEN: {size_bytes}>{maximum_bytes}"
        )


def parse_zenodo_record(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata")
    files = value.get("files")
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise CaversStage1Error("invalid Zenodo record schema")
    expected = {
        "id": ZENODO_RECORD_ID,
        "conceptrecid": ZENODO_CONCEPT_RECORD_ID,
        "doi": ZENODO_DOI,
        "revision": ZENODO_REVISION,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CaversStage1Error(f"Zenodo {key} mismatch")
    if metadata.get("version") != ZENODO_VERSION:
        raise CaversStage1Error("Zenodo version mismatch")
    if metadata.get("publication_date") != ZENODO_PUBLICATION_DATE:
        raise CaversStage1Error("Zenodo publication date mismatch")
    if value.get("status") != "published" or value.get("state") != "done":
        raise CaversStage1Error("Zenodo record is not a completed publication")
    if value.get("submitted") is not True:
        raise CaversStage1Error("Zenodo record is not submitted")
    license_value = metadata.get("license")
    if not isinstance(license_value, Mapping) or license_value.get("id") != ZENODO_LICENSE:
        raise CaversStage1Error("Zenodo dataset license mismatch")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise CaversStage1Error("invalid Zenodo file row")
        name = row.get("key")
        size = row.get("size")
        checksum = row.get("checksum")
        link = row.get("links", {}).get("self") if isinstance(row.get("links"), Mapping) else None
        if (
            not isinstance(name, str)
            or name in names
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(checksum, str)
            or not checksum.startswith("md5:")
            or not isinstance(link, str)
        ):
            raise CaversStage1Error("invalid or duplicate Zenodo file entry")
        names.add(name)
        rows.append(
            {
                "archive_name": name,
                "checksum": checksum,
                "download_url": link,
                "size_bytes": size,
            }
        )
    if len(rows) != 48:
        raise CaversStage1Error("Zenodo inventory must contain exactly 48 archives")
    total_archive_bytes = sum(row["size_bytes"] for row in rows)
    if total_archive_bytes != ZENODO_TOTAL_ARCHIVE_BYTES:
        raise CaversStage1Error("Zenodo inventory byte total mismatch")
    return {
        "created": value.get("created"),
        "doi": value["doi"],
        "file_count": len(rows),
        "files": sorted(rows, key=lambda row: row["archive_name"]),
        "license": license_value["id"],
        "modified": value.get("modified"),
        "publication_date": metadata.get("publication_date"),
        "record_id": value["id"],
        "revision": value["revision"],
        "title": metadata.get("title"),
        "total_archive_bytes": total_archive_bytes,
        "version": metadata["version"],
    }


class RangeClient:
    """Sequential retrying HTTP range client with an auditable byte budget."""

    def __init__(self, *, maximum_materialized_member_bytes: int) -> None:
        if maximum_materialized_member_bytes <= 0 or maximum_materialized_member_bytes > MAX_STAGE1_FILE_BYTES:
            raise Stage1LargeFileDownloadForbidden("invalid range materialization limit")
        self.maximum = maximum_materialized_member_bytes
        self.receipts: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        start: int,
        end: int,
        *,
        purpose: str,
        expected_total_bytes: int | None = None,
    ) -> bytes:
        if start < 0 or end < start:
            raise CaversStage1Error("invalid HTTP range")
        requested = end - start + 1
        if requested > self.maximum:
            raise Stage1LargeFileDownloadForbidden(
                f"STAGE1_LARGE_FILE_DOWNLOAD_FORBIDDEN range: {requested}>{self.maximum}"
            )
        last_error: Exception | None = None
        for attempt in range(6):
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "ZPRM-CAVERS-Stage1/1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    content_range = response.headers.get("Content-Range")
                    expected_content_range = (
                        f"bytes {start}-{end}/{expected_total_bytes}"
                        if expected_total_bytes is not None
                        else f"bytes {start}-{end}/"
                    )
                    if response.status != 206:
                        raise CaversStage1Error(
                            f"range server did not honor exact 206 request: {response.status}"
                        )
                    if content_range is None or (
                        content_range != expected_content_range
                        if expected_total_bytes is not None
                        else not content_range.startswith(expected_content_range)
                    ):
                        raise CaversStage1Error("range response Content-Range mismatch")
                    declared = response.headers.get("Content-Length")
                    if declared is None or int(declared) != requested:
                        raise CaversStage1Error("range response Content-Length mismatch or absent")
                    payload = response.read(requested + 1)
                    if len(payload) != requested:
                        raise CaversStage1Error(
                            f"range response body length mismatch: {len(payload)} != {requested}"
                        )
                    self.receipts.append(
                        {
                            "end_byte_inclusive": end,
                            "purpose": purpose,
                            "response_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "start_byte": start,
                            "status": 206,
                            "url": url,
                        }
                    )
                    return payload
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                TimeoutError,
                ConnectionError,
                CaversStage1Error,
            ) as error:
                last_error = error
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 5:
                    break
                delay = min(60, 2**attempt)
                if isinstance(error, urllib.error.HTTPError):
                    retry_after = error.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                time.sleep(delay)
        raise CaversStage1Error(f"range request failed after retries: {last_error}")

    def get_small_file(self, url: str, *, purpose: str, expected_bytes: int) -> bytes:
        if expected_bytes < 0 or expected_bytes > self.maximum:
            raise Stage1LargeFileDownloadForbidden("small-file endpoint exceeds Stage-1 limit")
        last_error: Exception | None = None
        for attempt in range(6):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ZPRM-CAVERS-Stage1/1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    if response.status != 200:
                        raise CaversStage1Error(
                            f"small-file endpoint did not return 200: {response.status}"
                        )
                    declared = response.headers.get("Content-Length")
                    if declared is not None:
                        declared_bytes = int(declared)
                        if declared_bytes > self.maximum:
                            raise Stage1LargeFileDownloadForbidden(
                                "STAGE1_LARGE_FILE_DOWNLOAD_FORBIDDEN by Content-Length"
                            )
                        if declared_bytes != expected_bytes:
                            raise CaversStage1Error("small-file Content-Length mismatch")
                    payload = response.read(expected_bytes + 1)
                    if len(payload) != expected_bytes:
                        raise CaversStage1Error(
                            f"small-file response size mismatch: {len(payload)} != {expected_bytes}"
                        )
                    self.receipts.append(
                        {
                            "end_byte_inclusive": None,
                            "purpose": purpose,
                            "response_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "start_byte": None,
                            "status": response.status,
                            "url": url,
                        }
                    )
                    return payload
            except Stage1LargeFileDownloadForbidden:
                raise
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                TimeoutError,
                ConnectionError,
                CaversStage1Error,
            ) as error:
                last_error = error
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 5:
                    break
                time.sleep(min(60, 2**attempt))
        raise CaversStage1Error(f"small-file request failed after retries: {last_error}")


def parse_zip_central_directory(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 46:
            raise CaversStage1Error("truncated ZIP central directory")
        values = struct.unpack_from("<4s6H3L5H2L", payload, offset)
        if values[0] != b"PK\x01\x02":
            raise CaversStage1Error("invalid ZIP central directory signature")
        filename_length, extra_length, comment_length = values[10], values[11], values[12]
        filename_bytes = payload[offset + 46 : offset + 46 + filename_length]
        filename = filename_bytes.decode("utf-8" if values[3] & 0x800 else "cp437")
        rows.append(
            {
                "compressed_size": int(values[8]),
                "compression_method": int(values[4]),
                "crc32": f"{values[7]:08x}",
                "local_header_offset": int(values[16]),
                "path": filename,
                "uncompressed_size": int(values[9]),
            }
        )
        offset += 46 + filename_length + extra_length + comment_length
    return rows


def read_remote_zip_index(
    client: RangeClient,
    *,
    archive_name: str,
    archive_size: int,
    url: str,
) -> dict[str, Any]:
    # A single bounded tail request normally contains both EOCD and the full
    # central directory.  This keeps the official API request count low.
    tail_size = min(4 * 1024 * 1024, archive_size)
    tail = client.get(
        url,
        archive_size - tail_size,
        archive_size - 1,
        purpose=f"{archive_name}:ZIP_EOCD_TAIL",
        expected_total_bytes=archive_size,
    )
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise CaversStage1Error(f"ZIP EOCD missing: {archive_name}")
    values = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    if values[1] != 0 or values[2] != 0 or values[3] != values[4]:
        raise CaversStage1Error("multi-disk ZIP is forbidden")
    entry_count, directory_size, directory_offset = values[4], values[5], values[6]
    if directory_size <= 0 or directory_size > 50_000_000:
        raise Stage1LargeFileDownloadForbidden("ZIP directory size is unsafe")
    tail_start = archive_size - tail_size
    if directory_offset >= tail_start and directory_offset + directory_size <= archive_size:
        relative = directory_offset - tail_start
        directory = tail[relative : relative + directory_size]
    else:
        directory = client.get(
            url,
            directory_offset,
            directory_offset + directory_size - 1,
            purpose=f"{archive_name}:ZIP_CENTRAL_DIRECTORY",
            expected_total_bytes=archive_size,
        )
    rows = parse_zip_central_directory(directory)
    if len(rows) != entry_count:
        raise CaversStage1Error("ZIP entry count mismatch")
    return {
        "archive_name": archive_name,
        "archive_size_bytes": archive_size,
        "central_directory_offset": directory_offset,
        "central_directory_sha256": hashlib.sha256(directory).hexdigest(),
        "central_directory_size": directory_size,
        "entry_count": entry_count,
        "entries": rows,
        "url": url,
    }


def extract_remote_zip_member(
    client: RangeClient,
    *,
    index: Mapping[str, Any],
    member: Mapping[str, Any],
) -> bytes:
    compressed_size = int(member["compressed_size"])
    uncompressed_size = int(member["uncompressed_size"])
    if compressed_size > client.maximum or uncompressed_size > client.maximum:
        raise Stage1LargeFileDownloadForbidden("remote ZIP member exceeds Stage-1 limit")
    method = int(member["compression_method"])
    if method not in (0, 8):
        raise CaversStage1Error(f"unsupported ZIP compression method: {method}")
    content_url = str(index["url"])
    if not content_url.endswith("/content"):
        raise CaversStage1Error("unexpected Zenodo content URL")
    member_url = (
        content_url[: -len("/content")]
        + "/container/"
        + urllib.parse.quote(str(member["path"]), safe="/")
    )
    payload = client.get_small_file(
        member_url,
        purpose=f"{index['archive_name']}:{member['path']}:CONTAINER_MEMBER",
        expected_bytes=uncompressed_size,
    )
    if len(payload) != uncompressed_size:
        raise CaversStage1Error("ZIP member uncompressed size mismatch")
    if f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}" != member["crc32"]:
        raise CaversStage1Error("ZIP member CRC mismatch")
    return payload


def _member_by_path(index: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    matches = [row for row in index["entries"] if row["path"] == path]
    if len(matches) != 1:
        raise CaversStage1Error(f"required ZIP member missing or duplicated: {path}")
    return matches[0]


def _verify_existing_member(path: Path, member: Mapping[str, Any]) -> None:
    payload = path.read_bytes()
    if len(payload) != member["uncompressed_size"]:
        raise CaversStage1Error(f"resume member size mismatch: {path}")
    if f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}" != member["crc32"]:
        raise CaversStage1Error(f"resume member CRC mismatch: {path}")


def _authenticate_sequence_checkpoint(
    *,
    sequence_id: str,
    inventory: Mapping[str, Mapping[str, Any]],
    source_root: Path,
) -> dict[str, Any] | None:
    inventory_path = source_root / "sequences" / sequence_id / "remote_zip_inventory.json"
    if not inventory_path.exists():
        return None
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    if value.get("sequence_id") != sequence_id:
        raise CaversStage1Error("resume sequence checkpoint identity mismatch")
    raw_name = f"{sequence_id}.zip"
    bag_name = f"{sequence_id}_rosbag.zip"
    raw = inventory.get(raw_name)
    bag = inventory.get(bag_name)
    if raw is None or bag is None:
        raise CaversStage1Error("resume checkpoint official archive is absent")
    for key, expected in (("raw_archive", raw), ("rosbag_archive", bag)):
        archive = value.get(key)
        if not isinstance(archive, Mapping) or (
            archive.get("archive_name") != expected["archive_name"]
            or archive.get("archive_size_bytes") != expected["size_bytes"]
            or archive.get("url") != expected["download_url"]
        ):
            raise CaversStage1Error("resume checkpoint archive provenance mismatch")
    selected = value.get("selected_members")
    if not isinstance(selected, list) or len(selected) != len(RAW_MEMBER_SUFFIXES) + 1:
        raise CaversStage1Error("resume checkpoint member inventory mismatch")
    expected_paths = {
        *(f"{sequence_id}/{suffix}" for suffix in RAW_MEMBER_SUFFIXES),
        f"{sequence_id}/{ROSBAG_MEMBER_SUFFIX}",
    }
    if {row.get("path") for row in selected if isinstance(row, Mapping)} != expected_paths:
        raise CaversStage1Error("resume checkpoint member allow-list mismatch")
    evidence_bytes = 0
    sequence_root = (source_root / "sequences" / sequence_id).resolve(strict=True)
    for row in selected:
        if not isinstance(row, Mapping):
            raise CaversStage1Error("invalid resume checkpoint member")
        local = Path(str(row.get("local_path", ""))).resolve(strict=True)
        if sequence_root not in local.parents or local.is_symlink():
            raise CaversStage1Error("resume checkpoint member path is unsafe")
        _verify_existing_member(local, row)
        if sha256_file(local) != row.get("sha256"):
            raise CaversStage1Error("resume checkpoint member SHA-256 mismatch")
        evidence_bytes += local.stat().st_size
    return {
        "evidence_bytes": evidence_bytes,
        "raw_archive_bytes_not_downloaded": raw["size_bytes"],
        "raw_archive_name": raw_name,
        "rosbag_archive_bytes_not_downloaded": bag["size_bytes"],
        "rosbag_archive_name": bag_name,
        "selected_member_count": len(selected),
        "sequence_id": sequence_id,
        "zip_inventory_path": str(inventory_path),
        "zip_inventory_sha256": sha256_file(inventory_path),
    }


def materialize_sequence_evidence(
    *,
    parsed_record: Mapping[str, Any],
    source_root: Path,
    maximum_bytes: int,
    resume_existing: bool = False,
) -> dict[str, Any]:
    inventory = {row["archive_name"]: row for row in parsed_record["files"]}
    client = RangeClient(maximum_materialized_member_bytes=maximum_bytes)
    sequence_rows: list[dict[str, Any]] = []
    resume_authenticated_sequence_ids: list[str] = []
    for sequence_id in CANDIDATE_SEQUENCES:
        if resume_existing:
            checkpoint = _authenticate_sequence_checkpoint(
                sequence_id=sequence_id,
                inventory=inventory,
                source_root=source_root,
            )
            if checkpoint is not None:
                sequence_rows.append(checkpoint)
                resume_authenticated_sequence_ids.append(sequence_id)
                continue
        raw_name = f"{sequence_id}.zip"
        bag_name = f"{sequence_id}_rosbag.zip"
        raw = inventory.get(raw_name)
        bag = inventory.get(bag_name)
        if raw is None or bag is None:
            raise CaversStage1Error(f"sequence archives missing: {sequence_id}")
        raw_index = read_remote_zip_index(
            client,
            archive_name=raw_name,
            archive_size=int(raw["size_bytes"]),
            url=str(raw["download_url"]),
        )
        bag_index = read_remote_zip_index(
            client,
            archive_name=bag_name,
            archive_size=int(bag["size_bytes"]),
            url=str(bag["download_url"]),
        )
        selected_members: list[dict[str, Any]] = []
        destination_root = source_root / "sequences" / sequence_id
        for suffix in RAW_MEMBER_SUFFIXES:
            member_path = f"{sequence_id}/{suffix}"
            member = _member_by_path(raw_index, member_path)
            destination = destination_root / suffix
            if destination.exists():
                _verify_existing_member(destination, member)
            else:
                atomic_write_bytes(
                    destination,
                    extract_remote_zip_member(client, index=raw_index, member=member),
                )
            selected_members.append(
                {
                    **dict(member),
                    "archive_name": raw_name,
                    "local_path": str(destination),
                    "sha256": sha256_file(destination),
                }
            )
        bag_member_path = f"{sequence_id}/{ROSBAG_MEMBER_SUFFIX}"
        bag_member = _member_by_path(bag_index, bag_member_path)
        bag_destination = destination_root / "rosbag_metadata.yaml"
        if bag_destination.exists():
            _verify_existing_member(bag_destination, bag_member)
        else:
            atomic_write_bytes(
                bag_destination,
                extract_remote_zip_member(client, index=bag_index, member=bag_member),
            )
        selected_members.append(
            {
                **dict(bag_member),
                "archive_name": bag_name,
                "local_path": str(bag_destination),
                "sha256": sha256_file(bag_destination),
            }
        )
        internal_inventory = {
            "raw_archive": {
                key: raw_index[key]
                for key in (
                    "archive_name",
                    "archive_size_bytes",
                    "central_directory_offset",
                    "central_directory_sha256",
                    "central_directory_size",
                    "entry_count",
                    "url",
                )
            },
            "rosbag_archive": {
                key: bag_index[key]
                for key in (
                    "archive_name",
                    "archive_size_bytes",
                    "central_directory_offset",
                    "central_directory_sha256",
                    "central_directory_size",
                    "entry_count",
                    "url",
                )
            },
            "selected_members": selected_members,
            "sequence_id": sequence_id,
        }
        inventory_path = destination_root / "remote_zip_inventory.json"
        atomic_write_json(inventory_path, internal_inventory)
        sequence_rows.append(
            {
                "evidence_bytes": sum(Path(row["local_path"]).stat().st_size for row in selected_members),
                "raw_archive_bytes_not_downloaded": raw["size_bytes"],
                "raw_archive_name": raw_name,
                "rosbag_archive_bytes_not_downloaded": bag["size_bytes"],
                "rosbag_archive_name": bag_name,
                "selected_member_count": len(selected_members),
                "sequence_id": sequence_id,
                "zip_inventory_path": str(inventory_path),
                "zip_inventory_sha256": sha256_file(inventory_path),
            }
        )
    receipt_path = source_root / "http_range_receipts.json"
    atomic_write_json(
        receipt_path,
        {
            "full_archive_download_count": 0,
            "maximum_materialized_member_bytes": maximum_bytes,
            "receipts": client.receipts,
            "resume_authenticated_sequence_count": len(resume_authenticated_sequence_ids),
            "resume_authenticated_sequence_ids": resume_authenticated_sequence_ids,
            "total_range_response_bytes": sum(row["response_bytes"] for row in client.receipts),
        },
    )
    return {
        "range_receipt_path": str(receipt_path),
        "range_receipt_sha256": sha256_file(receipt_path),
        "resume_authenticated_sequence_count": len(resume_authenticated_sequence_ids),
        "resume_authenticated_sequence_ids": resume_authenticated_sequence_ids,
        "sequences": sequence_rows,
        "total_range_response_bytes": sum(row["response_bytes"] for row in client.receipts),
    }


def load_materialized_sequence_evidence(
    *,
    parsed_record: Mapping[str, Any],
    source_root: Path,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Authenticate a completed small-file download checkpoint for resume."""

    receipt_path = source_root / "http_range_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("full_archive_download_count") != 0:
        raise CaversStage1Error("resume receipt claims a full archive download")
    if receipt.get("maximum_materialized_member_bytes") != maximum_bytes:
        raise CaversStage1Error("resume download limit mismatch")
    receipts = receipt.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise CaversStage1Error("resume HTTP receipt inventory is empty")
    if receipt.get("total_range_response_bytes") != sum(
        row.get("response_bytes", -1) for row in receipts if isinstance(row, Mapping)
    ):
        raise CaversStage1Error("resume HTTP receipt byte total mismatch")

    official = {row["archive_name"]: row for row in parsed_record["files"]}
    sequence_rows: list[dict[str, Any]] = []
    for sequence_id in CANDIDATE_SEQUENCES:
        inventory_path = source_root / "sequences" / sequence_id / "remote_zip_inventory.json"
        value = json.loads(inventory_path.read_text(encoding="utf-8"))
        if value.get("sequence_id") != sequence_id:
            raise CaversStage1Error("resume sequence inventory identity mismatch")
        raw_name = f"{sequence_id}.zip"
        bag_name = f"{sequence_id}_rosbag.zip"
        raw = official.get(raw_name)
        bag = official.get(bag_name)
        if raw is None or bag is None:
            raise CaversStage1Error("resume official archive identity missing")
        for key, expected in (("raw_archive", raw), ("rosbag_archive", bag)):
            archive = value.get(key)
            if not isinstance(archive, Mapping):
                raise CaversStage1Error("resume archive inventory missing")
            if (
                archive.get("archive_name") != expected["archive_name"]
                or archive.get("archive_size_bytes") != expected["size_bytes"]
                or archive.get("url") != expected["download_url"]
            ):
                raise CaversStage1Error("resume archive inventory provenance mismatch")
        selected = value.get("selected_members")
        if not isinstance(selected, list) or len(selected) != len(RAW_MEMBER_SUFFIXES) + 1:
            raise CaversStage1Error("resume selected-member inventory mismatch")
        evidence_bytes = 0
        for row in selected:
            if not isinstance(row, Mapping):
                raise CaversStage1Error("invalid resume selected-member row")
            path = Path(str(row.get("local_path", "")))
            expected_root = (source_root / "sequences" / sequence_id).resolve(strict=True)
            resolved = path.resolve(strict=True)
            if resolved.parent != expected_root and expected_root not in resolved.parents:
                raise CaversStage1Error("resume member path escapes its sequence root")
            _verify_existing_member(resolved, row)
            if sha256_file(resolved) != row.get("sha256"):
                raise CaversStage1Error("resume member SHA-256 mismatch")
            evidence_bytes += resolved.stat().st_size
        sequence_rows.append(
            {
                "evidence_bytes": evidence_bytes,
                "raw_archive_bytes_not_downloaded": raw["size_bytes"],
                "raw_archive_name": raw_name,
                "rosbag_archive_bytes_not_downloaded": bag["size_bytes"],
                "rosbag_archive_name": bag_name,
                "selected_member_count": len(selected),
                "sequence_id": sequence_id,
                "zip_inventory_path": str(inventory_path),
                "zip_inventory_sha256": sha256_file(inventory_path),
            }
        )
    return {
        "range_receipt_path": str(receipt_path),
        "range_receipt_sha256": sha256_file(receipt_path),
        "resume_authenticated_sequence_count": receipt.get(
            "resume_authenticated_sequence_count", 0
        ),
        "resume_authenticated_sequence_ids": receipt.get(
            "resume_authenticated_sequence_ids", []
        ),
        "sequences": sequence_rows,
        "total_range_response_bytes": receipt["total_range_response_bytes"],
    }


def _float(row: Mapping[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise CaversStage1Error(f"invalid numeric GT field {name}") from error


def audit_gt_csv(path: Path, sequence_id: str) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        required = (
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
        if any(name not in fieldnames for name in required):
            raise CaversStage1Error(f"GT schema lacks complete xyzw 6DoF fields: {sequence_id}")
        rows = list(reader)
    if len(rows) < 2:
        raise CaversStage1Error(f"GT has fewer than two rows: {sequence_id}")
    timestamps_raw = np.array([_float(row, "Timestamp") for row in rows], dtype=np.float64)
    timestamp_scale = 1e-9 if float(np.median(timestamps_raw)) > 1e12 else 1.0
    timestamps = timestamps_raw * timestamp_scale
    positions = np.array(
        [[_float(row, name) for name in ("PX", "PY", "PZ")] for row in rows], dtype=np.float64
    )
    quaternions = np.array(
        [[_float(row, name) for name in ("QX", "QY", "QZ", "QW")] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(timestamps).all() or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise CaversStage1Error("GT contains non-finite values")
    gaps = np.diff(timestamps)
    monotonic = bool(np.all(gaps > 0.0))
    norms = np.linalg.norm(quaternions, axis=1)
    quaternion_pass = bool(np.max(np.abs(norms - 1.0)) <= 1e-6)
    world_frames = sorted({row["Frame_ID"] for row in rows})
    child_frames = sorted({row["Child_Frame_ID"] for row in rows})
    first_rotation_angle = float(Rotation.from_quat(quaternions[0] / norms[0]).magnitude())
    local_reset = bool(np.linalg.norm(positions[0]) < 1e-6 and first_rotation_angle < 1e-6)
    velocity_fields = all(
        name in fieldnames for name in ("VX", "VY", "VZ", "VROLL", "VPITCH", "VYAW")
    )
    status = "PASS" if monotonic and quaternion_pass and len(world_frames) == 1 and len(child_frames) == 1 else "FAIL"
    report = {
        "child_frames": child_frames,
        "complete_6dof": True,
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "first_position_m": positions[0].tolist(),
        "first_quaternion_xyzw": quaternions[0].tolist(),
        "first_timestamp_s": float(timestamps[0]),
        "gt_sha256": sha256_file(path),
        "independent_reference_source": "OptiTrack /spaceuma/optitrack/odom",
        "last_timestamp_s": float(timestamps[-1]),
        "linear_and_angular_velocity_fields_present": velocity_fields,
        "max_quaternion_norm_error": float(np.max(np.abs(norms - 1.0))),
        "maximum_timestamp_gap_s": float(np.max(gaps)),
        "median_rate_hz": float(1.0 / np.median(gaps)),
        "position_finite": True,
        "quaternion_norm_pass": quaternion_pass,
        "quaternion_order": "qx,qy,qz,qw (scalar-last)",
        "row_count": len(rows),
        "sequence_id": sequence_id,
        "sequence_local_reset_detected": local_reset,
        "status": status,
        "strictly_monotonic": monotonic,
        "timestamp_scale_to_seconds": timestamp_scale,
        "world_frames": world_frames,
    }
    return report, np.column_stack((timestamps, positions))


def parse_tf_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output: list[dict[str, Any]] = []
    for csv_line_number, row in enumerate(rows, 2):
        translation = np.array([_float(row, name) for name in ("TX", "TY", "TZ")])
        raw_quaternion = np.array(
            [_float(row, name) for name in ("QX", "QY", "QZ", "QW")], dtype=np.float64
        )
        if not np.isfinite(translation).all() or not np.isfinite(raw_quaternion).all():
            raise CaversStage1Error("TF row contains non-finite values")
        raw_norm = float(np.linalg.norm(raw_quaternion))
        norm_error = abs(raw_norm - 1.0)
        if raw_norm == 0.0 or norm_error > TF_EXPORTED_QUATERNION_MAX_NORM_ERROR:
            raise CaversStage1Error("TF quaternion exceeds frozen export-rounding tolerance")
        quaternion = raw_quaternion / raw_norm
        transform = transform_from_xyzw(translation, quaternion)
        output.append(
            {
                "child_frame": row["Child_Frame_ID"],
                "csv_line_number": csv_line_number,
                "direction": f"T_{row['Frame_ID']}_{row['Child_Frame_ID']}",
                "parent_frame": row["Frame_ID"],
                "quaternion_xyzw": quaternion.tolist(),
                "quaternion_xyzw_raw": raw_quaternion.tolist(),
                "quaternion_raw_norm": raw_norm,
                "quaternion_raw_norm_error": norm_error,
                "quaternion_rounding_tolerance": TF_EXPORTED_QUATERNION_MAX_NORM_ERROR,
                "rotation_matrix": transform[:3, :3].tolist(),
                "timestamp": row.get("Timestamp", ""),
                "translation_m": translation.tolist(),
            }
        )
    return output


def select_rig_lidar_transform(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        dict(row)
        for row in rows
        if str(row["parent_frame"]) == "base_link"
        and str(row["child_frame"]) == "velodyne"
    ]
    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        identity = compact_sha256(
            {
                key: row[key]
                for key in (
                    "parent_frame",
                    "child_frame",
                    "translation_m",
                    "quaternion_xyzw",
                )
            }
        )
        unique[identity] = row
    if len(unique) != 1:
        return None
    selected = next(iter(unique.values()))
    t_parent_child = transform_from_xyzw(
        np.asarray(selected["translation_m"]), np.asarray(selected["quaternion_xyzw"])
    )
    composition_probe = compose_world_sensor(np.eye(4), t_parent_child)
    if not np.allclose(composition_probe, t_parent_child, rtol=0.0, atol=0.0):
        raise CaversStage1Error("T_map_rig @ T_rig_lidar direction test failed")
    selected["composition_rule"] = "T_map_lidar(t) = T_map_rig(t) @ T_rig_lidar"
    selected["direction_test_pass"] = True
    selected["units"] = {"rotation": "dimensionless", "translation": "m"}
    return selected


def audit_lidar_timestamps(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    timestamps_raw = np.array([_float(row, "Timestamp") for row in rows], dtype=np.float64)
    scale = 1e-9 if timestamps_raw.size and float(np.median(timestamps_raw)) > 1e12 else 1.0
    timestamps = timestamps_raw * scale
    monotonic = bool(timestamps.size > 1 and np.all(np.diff(timestamps) > 0.0))
    return {
        "first_timestamp_s": float(timestamps[0]) if timestamps.size else None,
        "last_timestamp_s": float(timestamps[-1]) if timestamps.size else None,
        "row_count": int(timestamps.size),
        "strictly_monotonic": monotonic,
        "timestamp_resolution_s": scale,
    }


def _topic_rows_from_metadata(path: Path) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    info = value.get("rosbag2_bagfile_information", {}) if isinstance(value, Mapping) else {}
    topics = info.get("topics_with_message_count", []) if isinstance(info, Mapping) else []
    output = []
    for row in topics:
        metadata = row.get("topic_metadata", {}) if isinstance(row, Mapping) else {}
        output.append(
            {
                "message_count": row.get("message_count"),
                "name": metadata.get("name"),
                "type": metadata.get("type"),
            }
        )
    return output


def _environment_report(repository: Path, data_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[f"{key}_bytes"] = int(value.strip().split()[0]) * 1024
    packages = {}
    for name in ("numpy", "scipy", "PyYAML", "pytest"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "collected_at_utc": utc_now(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total, "used_bytes": disk.used},
        "git": {
            "branch": run_text(["git", "branch", "--show-current"], cwd=repository).strip(),
            "commit": run_text(["git", "rev-parse", "HEAD"], cwd=repository).strip(),
            "tags": run_text(["git", "tag", "--list"], cwd=repository).splitlines(),
            "worktree_porcelain": run_text(["git", "status", "--porcelain"], cwd=repository).splitlines(),
        },
        "host": platform.node(),
        "machine": platform.machine(),
        "memory": memory,
        "os": platform.platform(),
        "packages": packages,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
    }


def _resume_finalizing_environment(
    *,
    repository: Path,
    runtime_root: Path,
    origin_environment: Mapping[str, Any],
    live_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Append immutable provenance for each implementation-changing resume."""

    origin_commit = origin_environment.get("git", {}).get("commit")
    live_commit = live_environment.get("git", {}).get("commit")
    if not isinstance(origin_commit, str) or re.fullmatch(r"[0-9a-f]{40}", origin_commit) is None:
        raise PermissionError("invalid resume origin commit")
    if not isinstance(live_commit, str) or re.fullmatch(r"[0-9a-f]{40}", live_commit) is None:
        raise PermissionError("invalid live resume commit")

    standard_path = runtime_root / "resume_environment_report.json"
    if not standard_path.is_file():
        first = {
            **live_environment,
            "resume_origin_commit": origin_commit,
            "resume_policy": RESUME_POLICY,
        }
        atomic_write_json(standard_path, first)
        return first

    paths = [standard_path, *sorted(runtime_root.glob("resume_environment_report_*.json"))]
    reports_by_commit: dict[str, dict[str, Any]] = {}
    standard_commit: str | None = None
    for index, path in enumerate(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        commit = report.get("git", {}).get("commit")
        if (
            not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or report.get("git", {}).get("branch") != EXPECTED_BRANCH
            or report.get("git", {}).get("worktree_porcelain") != []
            or report.get("resume_origin_commit") != origin_commit
            or report.get("resume_policy") != RESUME_POLICY
            or commit in reports_by_commit
        ):
            raise PermissionError("invalid immutable resume environment report")
        if index == 0:
            standard_commit = commit
        elif path.name != f"resume_environment_report_{commit}.json":
            raise PermissionError("resume environment filename/commit mismatch")
        reports_by_commit[commit] = report

    for commit in reports_by_commit:
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", origin_commit, commit],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode or subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, live_commit],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            raise PermissionError("resume environment commit is outside the active lineage")

    assert standard_commit is not None
    consumed = {standard_commit}
    latest_commit = standard_commit
    latest = reports_by_commit[latest_commit]
    while True:
        successors = [
            (commit, report)
            for commit, report in reports_by_commit.items()
            if commit not in consumed and report.get("previous_resume_commit") == latest_commit
        ]
        if not successors:
            break
        if len(successors) != 1:
            raise PermissionError("resume provenance chain forks")
        successor_commit, successor = successors[0]
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", latest_commit, successor_commit],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            raise PermissionError("resume provenance link is not forward-only")
        latest_commit, latest = successor_commit, successor
        consumed.add(latest_commit)
    if consumed != set(reports_by_commit):
        raise PermissionError("resume provenance chain is disconnected")
    if live_commit == latest_commit:
        return latest
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", latest_commit, live_commit],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise PermissionError("resume implementation lineage is not forward-only")
    current = {
        **live_environment,
        "previous_resume_commit": latest_commit,
        "resume_origin_commit": origin_commit,
        "resume_policy": RESUME_POLICY,
    }
    atomic_write_json(runtime_root / f"resume_environment_report_{live_commit}.json", current)
    return current


def _source_files(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".git" in path.parts:
            continue
        rows.append(
            {
                "local_path": str(path),
                "relative_path": path.relative_to(data_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "status": "MATERIALIZED_SMALL_STAGE1_EVIDENCE",
            }
        )
    return rows


def _cleanup_report() -> dict[str, Any]:
    return {
        "allowed_deleted_root": "/home/lj/zero_perturbation_data/rts_gt_v1",
        "deleted_root_absent": not Path("/home/lj/zero_perturbation_data/rts_gt_v1").exists(),
        "deleted_root_allocated_bytes_before": 232075264,
        "filesystem_available_bytes_after": 111142998016,
        "filesystem_available_bytes_before": 110910959616,
        "filesystem_available_bytes_delta": 232038400,
        "rts_failure_frozen_sha256sums": str(
            Path("/home/lj/ZPRM/frozen_assets/real_data_rts_gt_stage1_failed_v1/SHA256SUMS")
        ),
        "rts_runtime_preserved": Path(
            "/home/lj/zero_perturbation_runtime/real_data/rts_gt_single_dataset_v1"
        ).is_dir(),
        "synthetic_runtime_preserved": Path(
            "/home/lj/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3_requalified"
        ).is_dir(),
    }


def _uncertainty_rows(reference_sha: str, extrinsic_sha: str, time_sha: str) -> list[dict[str, Any]]:
    values = (
        ("OptiTrack position uncertainty", "UNKNOWN", "m", "UNKNOWN", "QUALITATIVE_ONLY", "paper qualitative accuracy only", reference_sha, "UNKNOWN"),
        ("OptiTrack orientation uncertainty", "UNKNOWN", "rad", "UNKNOWN", "ABSENT_NUMERIC_BOUND", "no numeric release field", reference_sha, "UNKNOWN"),
        ("GT timestamp uncertainty", "UNKNOWN", "s", "UNKNOWN", "ABSENT_NUMERIC_BOUND", "ROS header timestamp without accuracy bound", time_sha, "UNKNOWN"),
        ("LiDAR timestamp uncertainty", "UNKNOWN", "s", "UNKNOWN", "ABSENT_NUMERIC_BOUND", "sensor/header timestamp without accuracy bound", time_sha, "UNKNOWN"),
        ("synchronization/delay uncertainty", "UNKNOWN", "s", "UNKNOWN", "ABSENT_NUMERIC_BOUND", "no hardware synchronization and no numeric delay", time_sha, "UNKNOWN"),
        ("CAD extrinsic translation uncertainty", "UNKNOWN", "m", "UNKNOWN", "NOMINAL_ONLY", "nominal transform is not an uncertainty", extrinsic_sha, "UNKNOWN"),
        ("CAD extrinsic rotation uncertainty", "UNKNOWN", "rad", "UNKNOWN", "NOMINAL_ONLY", "nominal transform is not an uncertainty", extrinsic_sha, "UNKNOWN"),
        ("interpolation uncertainty", "UNKNOWN", "m/rad", "UNKNOWN", "NOT_YET_MATERIALIZED", "future method candidate only", reference_sha, "UNKNOWN"),
        ("future map accumulation uncertainty", "UNKNOWN", "m", "UNKNOWN", "NOT_YET_MATERIALIZED", "map not materialized in Stage-1", "NOT_AVAILABLE", "UNKNOWN"),
        ("motion distortion / deskew uncertainty", "UNKNOWN", "m", "UNKNOWN", "NOT_YET_MATERIALIZED", "LiDAR not downloaded in Stage-1", "NOT_AVAILABLE", "UNKNOWN"),
    )
    return [
        {
            "component": component,
            "evidence_type": evidence_type,
            "source": source,
            "source_sha256": source_sha,
            "status": status,
            "uncertainty_type": uncertainty_type,
            "unit": unit,
            "value": value,
        }
        for component, value, unit, uncertainty_type, evidence_type, source, source_sha, status in values
    ]


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# CAVERS 单数据集 Stage-1 总结", ""]
    for index, answer in enumerate(summary["answers"], 1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## 最终结论",
            "",
            f"`CAVERS_STAGE1_READY={str(summary['cavers_stage1_ready']).lower()}`",
            "",
            summary["final_conclusion"],
            "",
            "`weak_snapshot_count=0`，`rich_snapshot_count=0`，`snapshot_count=0`，`planned_future_trials=0`，`registration_execution_count=0`。",
            "",
            "`REAL_DATA_RUN_AUTHORIZED=false`，`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256sums_payload(root: Path, names: Iterable[str]) -> bytes:
    return "".join(f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)).encode("utf-8")


def execute_cavers_stage1(
    *,
    repository: Path,
    data_root: Path,
    runtime_root: Path,
    mode: str,
    metadata_only: bool,
    maximum_single_download_bytes: int,
    no_registration: bool,
) -> dict[str, Any]:
    repository_argument = Path(repository)
    data_root_argument = Path(data_root)
    runtime_root_argument = Path(runtime_root)
    repository = repository_argument.resolve(strict=True)
    data_root = data_root_argument.resolve(strict=True)
    source_root = (data_root / "source_metadata").resolve(strict=True)
    runtime_root = runtime_root_argument.resolve(strict=False)
    if not all(path.is_absolute() for path in (repository_argument, data_root_argument, runtime_root_argument)):
        raise ValueError("repository, data and runtime roots must be absolute")
    if data_root_argument != data_root or runtime_root_argument != runtime_root:
        raise ValueError("data and runtime roots must be canonical and may not be symlinks")
    if repository == data_root or repository in data_root.parents or data_root in repository.parents:
        raise ValueError("data root and repository must be disjoint")
    if repository == runtime_root or repository in runtime_root.parents or runtime_root in repository.parents:
        raise ValueError("runtime root and repository must be disjoint")
    if data_root == runtime_root or data_root in runtime_root.parents or runtime_root in data_root.parents:
        raise ValueError("data and runtime roots must be disjoint")
    if mode not in ("fresh", "resume"):
        raise ValueError("mode must be fresh or resume")
    if metadata_only is not True or no_registration is not True:
        raise PermissionError("--metadata-only and --no-registration are mandatory")
    if maximum_single_download_bytes > MAX_STAGE1_FILE_BYTES or maximum_single_download_bytes <= 0:
        raise Stage1LargeFileDownloadForbidden("invalid max-single-download-bytes")
    if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise PermissionError("ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is mandatory")
    if run_text(["git", "branch", "--show-current"], cwd=repository).strip() != EXPECTED_BRANCH:
        raise PermissionError("CAVERS preparation branch mismatch")
    if run_text(["git", "status", "--porcelain"], cwd=repository).strip():
        raise PermissionError("formal CAVERS Stage-1 requires clean worktree")
    for ancestor in (
        "01b19b1ba89376cb53aa082b77032b562888eb7b",
        "862aa44ca8bb8f2b98cdea45b76b13ea21875c9a",
    ):
        if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], cwd=repository).returncode:
            raise PermissionError(f"required history ancestor missing: {ancestor}")
    if mode == "resume" and not runtime_root.is_dir():
        raise FileNotFoundError("resume runtime is absent")
    if mode == "resume" and (runtime_root / "SHA256SUMS").is_file():
        if not (runtime_root / "cavers_stage1_manifest.json").is_file():
            raise CaversStage1Error("resume checksum exists without formal manifest")
        from .cavers_stage1_verifier import verify_cavers_stage1

        verification = verify_cavers_stage1(
            repository=repository, data_root=data_root, runtime_root=runtime_root
        )
        return verification["summary"]
    if mode == "fresh":
        if os.path.lexists(runtime_root):
            raise FileExistsError("fresh runtime root must be absent")
        runtime_root.mkdir(parents=True)

    environment_path = runtime_root / "environment_report.json"
    if mode == "resume" and environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if (
            environment.get("git", {}).get("branch") != EXPECTED_BRANCH
            or environment.get("git", {}).get("worktree_porcelain") != []
        ):
            raise PermissionError("resume environment identity mismatch")
        live_environment = _environment_report(repository, data_root)
        if live_environment["git"]["worktree_porcelain"]:
            raise PermissionError("worktree changed before resume")
        finalizing_environment = _resume_finalizing_environment(
            repository=repository,
            runtime_root=runtime_root,
            origin_environment=environment,
            live_environment=live_environment,
        )
    else:
        environment = _environment_report(repository, data_root)
        if environment["git"]["worktree_porcelain"]:
            raise PermissionError("worktree changed during preflight")
        atomic_write_json(environment_path, environment)
        finalizing_environment = environment
    cleanup = _cleanup_report()
    atomic_write_json(runtime_root / "cleanup_report.json", cleanup)
    atomic_write_json(runtime_root / "cavers_cleanup_report.json", cleanup)

    raw_record_path = source_root / "zenodo_record_metadata.json"
    if sha256_file(raw_record_path) != ZENODO_METADATA_SHA256:
        raise CaversStage1Error("Zenodo metadata JSON SHA-256 mismatch")
    raw_record = json.loads(raw_record_path.read_text(encoding="utf-8"))
    parsed_record = parse_zenodo_record(raw_record)
    atomic_write_bytes(runtime_root / "zenodo_record_metadata.json", raw_record_path.read_bytes())
    inventory_fields = ("archive_name", "size_bytes", "checksum", "download_url")
    atomic_write_csv(runtime_root / "zenodo_file_inventory.csv", parsed_record["files"], inventory_fields)

    github_repository = source_root / "github_repository"
    github_commit = run_text(["git", "rev-parse", "HEAD"], cwd=github_repository).strip()
    if github_commit != GITHUB_COMMIT or run_text(["git", "status", "--porcelain"], cwd=github_repository).strip():
        raise PermissionError("official GitHub source is not the pinned clean commit")
    github_files = [
        {
            "path": path.relative_to(github_repository).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(github_repository.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    ]
    github_manifest = {
        "code_license": GITHUB_CODE_LICENSE,
        "commit": github_commit,
        "files": github_files,
        "repository_url": GITHUB_URL,
        "tree_rows_sha256": compact_sha256(github_files),
    }
    atomic_write_json(runtime_root / "github_source_manifest.json", github_manifest)
    paper_path = source_root / f"cavers_paper_{ARXIV_ID}.pdf"
    if sha256_file(paper_path) != ARXIV_PDF_SHA256:
        raise CaversStage1Error("official arXiv PDF SHA-256 mismatch")
    license_report = (
        "# CAVERS license and citation\n\n"
        f"- Dataset: `{ZENODO_DOI}`, exact record `{ZENODO_RECORD_ID}`, version `{ZENODO_VERSION}`, revision `{ZENODO_REVISION}`.\n"
        "- Dataset license: CC BY 4.0 (`cc-by-4.0`).\n"
        f"- Helper code: GitHub commit `{GITHUB_COMMIT}`, MIT license.\n"
        f"- Paper: arXiv:{ARXIV_ID}.\n\n"
        "Dataset and code licenses are recorded separately and are not treated as interchangeable.\n"
    )
    atomic_write_bytes(runtime_root / "license_and_citation_report.md", license_report.encode("utf-8"))

    with NoRegistrationGuard(open3d_module=None) as guard:
        static_audit = assert_preparation_sources_are_safe(
            repository / "src/phase_a_harness/real_data_preparation"
        )
        receipt_path = source_root / "http_range_receipts.json"
        if mode == "resume" and receipt_path.is_file():
            evidence = load_materialized_sequence_evidence(
                parsed_record=parsed_record,
                source_root=source_root,
                maximum_bytes=maximum_single_download_bytes,
            )
        else:
            evidence = materialize_sequence_evidence(
                parsed_record=parsed_record,
                source_root=source_root,
                maximum_bytes=maximum_single_download_bytes,
                resume_existing=mode == "resume",
            )
        official_source = {
            "arxiv_id": ARXIV_ID,
            "arxiv_pdf": {
                "path": str(paper_path),
                "sha256": ARXIV_PDF_SHA256,
                "url": f"https://arxiv.org/pdf/{ARXIV_ID}",
            },
            "dataset_license": ZENODO_LICENSE,
            "github_code_license": GITHUB_CODE_LICENSE,
            "github_commit": GITHUB_COMMIT,
            "github_url": GITHUB_URL,
            "large_archive_download_count": 0,
            "record": {key: parsed_record[key] for key in parsed_record if key != "files"},
            "source_evidence": evidence,
            "static_no_registration_audit": static_audit,
            "zenodo_api_url": ZENODO_API_URL,
            "zenodo_metadata_sha256": ZENODO_METADATA_SHA256,
        }
        atomic_write_json(runtime_root / "official_source_manifest.json", official_source)

        reference_rows: list[dict[str, Any]] = []
        trajectories: dict[str, np.ndarray] = {}
        extrinsics: dict[str, dict[str, Any] | None] = {}
        time_rows: list[dict[str, Any]] = []
        sequence_inventory: list[dict[str, Any]] = []
        for sequence_id in CANDIDATE_SEQUENCES:
            sequence_root = source_root / "sequences" / sequence_id
            reference, trajectory = audit_gt_csv(
                sequence_root / "GT_ODOM/data.csv", sequence_id
            )
            reference_rows.append(reference)
            trajectories[sequence_id] = trajectory
            static_rows = parse_tf_csv(sequence_root / "TF_STATIC/data.csv")
            selected_extrinsic = select_rig_lidar_transform(static_rows)
            if selected_extrinsic is not None:
                selected_extrinsic.update(
                    {
                        "sequence_id": sequence_id,
                        "source_file": str(sequence_root / "TF_STATIC/data.csv"),
                        "source_sha256": sha256_file(sequence_root / "TF_STATIC/data.csv"),
                        "source_topic": "/tf_static",
                    }
                )
            lidar_time = audit_lidar_timestamps(sequence_root / "VELODYNE_CLOUD/data.csv")
            topics = _topic_rows_from_metadata(sequence_root / "rosbag_metadata.yaml")
            topic_by_name = {row["name"]: row for row in topics}
            optitrack_topic = topic_by_name.get("/spaceuma/optitrack/odom", {})
            optitrack_topic_present = (
                optitrack_topic.get("type") == "nav_msgs/msg/Odometry"
                and isinstance(optitrack_topic.get("message_count"), int)
                and optitrack_topic["message_count"] > 0
            )
            tf_static_topic = topic_by_name.get("/tf_static", {})
            tf_static_topic_present = (
                tf_static_topic.get("type") == "tf2_msgs/msg/TFMessage"
                and isinstance(tf_static_topic.get("message_count"), int)
                and tf_static_topic["message_count"] > 0
            )
            velodyne_topic = topic_by_name.get("/spaceuma/velodyne_points", {})
            velodyne_topic_present = (
                velodyne_topic.get("type") == "sensor_msgs/msg/PointCloud2"
                and isinstance(velodyne_topic.get("message_count"), int)
                and velodyne_topic["message_count"] > 0
            )
            reference["rosbag_optitrack_topic_present"] = optitrack_topic_present
            reference["rosbag_optitrack_message_count"] = optitrack_topic.get("message_count", 0)
            reference["status"] = (
                "PASS" if reference["status"] == "PASS" and optitrack_topic_present else "FAIL"
            )
            if not tf_static_topic_present:
                selected_extrinsic = None
            extrinsics[sequence_id] = selected_extrinsic
            gt_start, gt_end = reference["first_timestamp_s"], reference["last_timestamp_s"]
            lidar_start, lidar_end = lidar_time["first_timestamp_s"], lidar_time["last_timestamp_s"]
            overlap_start = max(gt_start, lidar_start) if lidar_start is not None else None
            overlap_end = min(gt_end, lidar_end) if lidar_end is not None else None
            span_overlap = max(0.0, overlap_end - overlap_start) if overlap_start is not None else 0.0
            time_rows.append(
                {
                    "clock_domains": ["ROS2 recording timestamp", "message header timestamp", "sensor timestamp"],
                    "documented_processing_delay_s": "UNKNOWN",
                    "documented_sensor_delay_s": "UNKNOWN",
                    "gt_lidar_span_overlap_s": span_overlap,
                    "hardware_sync_available": False,
                    "lidar_timestamp": lidar_time,
                    "maximum_gt_gap_s": reference["maximum_timestamp_gap_s"],
                    "measured_metadata_offset_s": abs(gt_start - lidar_start) if lidar_start is not None else "UNKNOWN",
                    "measured_metadata_offset_interpretation": "GT/LiDAR exported span-start difference; not a sensor-delay estimate",
                    "sequence_id": sequence_id,
                    "shared_recording_clock": True,
                    "tf_static_topic_present": tf_static_topic_present,
                    "time_sync_uncertainty": "UNKNOWN",
                    "timestamp_resolution_s": lidar_time["timestamp_resolution_s"],
                    "velodyne_topic_present": velodyne_topic_present,
                }
            )
            sequence_inventory.append(
                {
                    "gt_available": reference["status"] == "PASS",
                    "gt_duration_s": reference["duration_s"],
                    "gt_rate_hz": reference["median_rate_hz"],
                    "platform_type": "DIABLO_ROBOT" if "diablo" in sequence_id else "HANDHELD_RIG",
                    "reference_frame": ";".join(reference["world_frames"]),
                    "rig_velodyne_transform_available": selected_extrinsic is not None,
                    "room": "SALA_DEL_DIABLO",
                    "sequence_id": sequence_id,
                    "timestamp_index_available": lidar_time["row_count"] > 0,
                    "transform_calibration_present": bool(static_rows) and tf_static_topic_present,
                    "velodyne_present": velodyne_topic_present and lidar_time["row_count"] > 0,
                }
            )

        atomic_write_csv(
            runtime_root / "cavers_reference_audit.csv",
            reference_rows,
            REFERENCE_AUDIT_FIELDS,
        )
        reference_audit = {
            "candidate_count": len(reference_rows),
            "independent_optitrack_6dof_pass_count": sum(row["status"] == "PASS" for row in reference_rows),
            "quaternion_scalar_last": True,
            "rows": reference_rows,
            "status": "PASS" if all(row["status"] == "PASS" for row in reference_rows) else "FAIL",
        }
        atomic_write_json(runtime_root / "cavers_reference_audit.json", reference_audit)

        sequence_fields = (
            "sequence_id", "room", "platform_type", "gt_available", "gt_duration_s", "gt_rate_hz",
            "velodyne_present", "transform_calibration_present", "timestamp_index_available",
            "reference_frame", "rig_velodyne_transform_available",
        )
        atomic_write_csv(runtime_root / "cavers_sequence_inventory.csv", sequence_inventory, sequence_fields)
        atomic_write_json(
            runtime_root / "cavers_sequence_inventory.json",
            {
                "candidate_sequences": list(CANDIDATE_SEQUENCES),
                "excluded_no_independent_gt": list(EXCLUDED_NO_GT_SEQUENCES),
                "rows": sequence_inventory,
            },
        )

        world_frames = sorted({frame for row in reference_rows for frame in row["world_frames"]})
        reset_sequences = sorted(
            row["sequence_id"] for row in reference_rows if row["sequence_local_reset_detected"]
        )
        same_named_nonreset_frame = (
            reference_audit["status"] == "PASS"
            and len(world_frames) == 1
            and not reset_sequences
            and all(row["independent_reference_source"].startswith("OptiTrack") for row in reference_rows)
        )
        # The release does not publish a shared calibration/session identifier
        # or an explicit guarantee that the mocap base was unchanged across all
        # twelve recordings.  A common string "map" and non-zero first poses
        # are useful negative-reset evidence, but are not sufficient proof.
        shared_calibration_session_proven = False
        shared_world = same_named_nonreset_frame and shared_calibration_session_proven
        common_rows = [
            {
                "common_world_frame_status": "PASS" if shared_world else "FAIL",
                "sequence_id": row["sequence_id"],
                "sequence_local_reset_detected": row["sequence_local_reset_detected"],
                "shared_calibration_evidence": "NOT_PUBLISHED: no cross-recording calibration/session ID",
                "world_frame": ";".join(row["world_frames"]),
                "world_frame_source": row["gt_sha256"],
            }
            for row in reference_rows
        ]
        common_world = {
            "common_world_frame_status": "PASS" if shared_world else "FAIL",
            "evidence": [
                "Direct independent OptiTrack odometry is exported from one named topic.",
                "All candidate GT files use one identical Frame_ID.",
                "First-pose reset detection is evaluated numerically and must be false for every sequence.",
                "No trajectory or point-cloud fitting is used.",
            ],
            "registration_or_trajectory_alignment_used": False,
            "rows": common_rows,
            "same_named_nonreset_frame_observed": same_named_nonreset_frame,
            "shared_calibration_session_proven": shared_calibration_session_proven,
            "sequence_local_reset_detected": bool(reset_sequences),
            "sequence_local_reset_ids": reset_sequences,
            "unique_world_frames": world_frames,
        }
        atomic_write_json(runtime_root / "cavers_common_world_frame_audit.json", common_world)

        extrinsic_rows = []
        for sequence_id in CANDIDATE_SEQUENCES:
            selected = extrinsics[sequence_id]
            extrinsic_rows.append(
                {
                    "sequence_id": sequence_id,
                    "status": "PASS" if selected else "FAIL",
                    "transform": selected,
                }
            )
        extrinsic_identities = {
            compact_sha256(
                {
                    key: row["transform"][key]
                    for key in ("parent_frame", "child_frame", "translation_m", "quaternion_xyzw")
                }
            )
            for row in extrinsic_rows
            if row["transform"] is not None
        }
        extrinsic_status = (
            "PASS"
            if all(row["status"] == "PASS" for row in extrinsic_rows)
            and len(extrinsic_identities) == 1
            else "FAIL"
        )
        extrinsic_audit = {
            "extrinsic_uncertainty": "UNKNOWN",
            "identical_cross_sequence_transform_count": len(extrinsic_identities),
            "nominal_cad_transform_is_zero_uncertainty": False,
            "rows": extrinsic_rows,
            "status": extrinsic_status,
        }
        atomic_write_json(runtime_root / "cavers_extrinsic_audit.json", extrinsic_audit)
        transform_chain = {
            "composition": "T_map_lidar(t) = T_map_rig(t) @ T_rig_lidar",
            "direct_optitrack_gt": True,
            "extrinsics": extrinsics,
            "status": extrinsic_status,
        }
        atomic_write_json(runtime_root / "cavers_transform_chain_manifest.json", transform_chain)

        time_status = "PASS_WITH_DOCUMENTED_LIMITATION" if all(
            row["gt_lidar_span_overlap_s"] > 0.0
            and row["lidar_timestamp"]["strictly_monotonic"]
            and row["velodyne_topic_present"]
            for row in time_rows
        ) else "FAIL"
        time_audit = {
            "candidate_interpolation_method": "linear translation plus quaternion SLERP, gap-limited",
            "hardware_level_synchronization": False,
            "rows": time_rows,
            "status": time_status,
            "time_sync_uncertainty_status": "UNKNOWN",
            "zero_delay_assumed": False,
        }
        atomic_write_json(runtime_root / "cavers_time_sync_audit.json", time_audit)

        reference_sha = sha256_file(runtime_root / "cavers_reference_audit.json")
        extrinsic_sha = sha256_file(runtime_root / "cavers_extrinsic_audit.json")
        time_sha = sha256_file(runtime_root / "cavers_time_sync_audit.json")
        uncertainty_rows = _uncertainty_rows(reference_sha, extrinsic_sha, time_sha)
        if any(row["uncertainty_type"] not in ALLOWED_UNCERTAINTY_TYPES for row in uncertainty_rows):
            raise CaversStage1Error("invalid uncertainty type")
        uncertainty_status = "PARTIAL" if reference_audit["status"] == "PASS" else "FAIL"
        uncertainty_fields = (
            "component", "value", "unit", "uncertainty_type", "evidence_type", "source",
            "source_sha256", "status",
        )
        atomic_write_csv(
            runtime_root / "cavers_stage1_uncertainty_feasibility.csv",
            uncertainty_rows,
            uncertainty_fields,
        )
        uncertainty = {
            "all_unknown_values_preserved_as_unknown": True,
            "nominal_transform_treated_as_zero_uncertainty": False,
            "rows": uncertainty_rows,
            "status": uncertainty_status,
        }
        atomic_write_json(
            runtime_root / "cavers_stage1_uncertainty_feasibility.json", uncertainty
        )

        transform_gate = extrinsic_status == "PASS"
        overlap_gate = reference_audit["status"] == "PASS" and shared_world and transform_gate
        pair_rows: list[dict[str, Any]] = []
        if overlap_gate:
            for map_id in CANDIDATE_SEQUENCES:
                for query_id in CANDIDATE_SEQUENCES:
                    if map_id == query_id:
                        continue
                    result = compute_stage1_gt_overlap(
                        trajectories[map_id],
                        trajectories[query_id],
                        common_world_frame_proven=True,
                    )
                    pair_rows.append(
                        {
                            **result,
                            "common_world_frame_proven": True,
                            "map_independent_6dof": True,
                            "map_rig_lidar_transform_available": extrinsics[map_id] is not None,
                            "map_sequence_id": map_id,
                            "query_independent_6dof": True,
                            "query_rig_lidar_transform_available": extrinsics[query_id] is not None,
                            "query_sequence_id": query_id,
                        }
                    )
        ranked = rank_distinct_stage1_pairs(pair_rows)
        pair_selection = {
            "PRIMARY_CANDIDATE_MAP_QUERY_PAIR": ranked[0] if len(ranked) > 0 else None,
            "RESERVE_PAIR_1": ranked[1] if len(ranked) > 1 else None,
            "RESERVE_PAIR_2": ranked[2] if len(ranked) > 2 else None,
            "activation_policy": "reserve only for corruption, missing GT, download/checksum, or preregistered infrastructure failure; never registration outcome",
            "eligible_pair_count": len(ranked),
            "ranking_rule": [
                "covered duration descending",
                "coverage fraction descending",
                "eligible 5s intervals descending",
                "nearest distance q95 ascending",
                "map ID lexicographic",
                "query ID lexicographic",
            ],
        }
        overlap_csv_rows = [
            {
                "map_sequence_id": row["map_sequence_id"],
                "query_sequence_id": row["query_sequence_id"],
                "overlap_status": row["overlap_status"],
                "total_covered_duration_s": row["total_covered_duration_s"],
                "coverage_fraction": row["coverage_fraction"],
                "eligible_nonoverlapping_5s_interval_count": row[
                    "eligible_nonoverlapping_5s_interval_count"
                ],
                "nearest_distance_median_m": row["nearest_distance_median_m"],
                "nearest_distance_q95_m": row["nearest_distance_q95_m"],
                "nearest_distance_max_m": row["nearest_distance_max_m"],
            }
            for row in pair_rows
        ]
        overlap_fields = (
            "map_sequence_id", "query_sequence_id", "overlap_status", "total_covered_duration_s",
            "coverage_fraction", "eligible_nonoverlapping_5s_interval_count",
            "nearest_distance_median_m", "nearest_distance_q95_m", "nearest_distance_max_m",
        )
        atomic_write_csv(runtime_root / "cavers_gt_only_overlap_matrix.csv", overlap_csv_rows, overlap_fields)
        atomic_write_json(
            runtime_root / "cavers_gt_only_overlap_matrix.json",
            {
                "contract": asdict(FROZEN_STAGE1_OVERLAP_CONTRACT),
                "gate_status": "PASS" if overlap_gate else "NOT_COMPUTABLE",
                "pair_rows": pair_rows,
                "point_cloud_or_registration_consulted": False,
            },
        )
        atomic_write_json(runtime_root / "cavers_pair_selection.json", pair_selection)

        backend_hash = sha256_file(repository / "frozen_assets/backend_parameter_contract.json")
        r02 = "PASS" if reference_audit["status"] == "PASS" else "FAIL"
        r03 = "PASS_CANDIDATE" if ranked else "BLOCKED_STAGE1"
        r04 = "PASS" if ranked and shared_world else "FAIL"
        r05 = (
            "PASS_WITH_DOCUMENTED_LIMITATION"
            if extrinsic_status == "PASS" and time_status == "PASS_WITH_DOCUMENTED_LIMITATION"
            else "FAIL"
        )
        overlap_status = "PASS" if ranked else "FAIL" if overlap_gate else "NOT_COMPUTABLE"
        ready = (
            r02 == "PASS"
            and r03 == "PASS_CANDIDATE"
            and r04 == "PASS"
            and r05 in ("PASS", "PASS_WITH_DOCUMENTED_LIMITATION")
            and overlap_status == "PASS"
            and uncertainty_status == "PASS_FEASIBLE"
        )
        eligibility = {
            "CAVERS_STAGE1_READY": ready,
            "GT_ONLY_OVERLAP": overlap_status,
            "R01": "PENDING_SECOND_DATASET",
            "R02": r02,
            "R03": r03,
            "R04": r04,
            "R05": r05,
            "R06": "BLOCKED_STAGE1",
            "R07": "BLOCKED_STAGE1",
            "R08": "BLOCKED_STAGE1",
            "R09": "PASS" if backend_hash == BACKEND_PARAMETER_SHA256 else "FAIL",
            "R10": uncertainty_status,
            "R14": "BLOCKED_STAGE1",
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "REAL_DATA_RUN_AUTHORIZED": False,
            "SINGLE_DATASET_PREREGISTRATION_READY": False,
            "large_lidar_download_count": 0,
            "planned_future_trials": 0,
            "primary_hard_blockers": [
                key
                for key, failed in (
                    ("COMMON_WORLD_FRAME", not shared_world),
                    ("R05_TRANSFORM_TIME_CHAIN", r05 == "FAIL"),
                    ("GT_ONLY_OVERLAP", overlap_status != "PASS"),
                    ("R10_UNCERTAINTY_FEASIBILITY", uncertainty_status != "PASS_FEASIBLE"),
                )
                if failed
            ],
            "registration_execution_count": 0,
            "rich_snapshot_count": 0,
            "snapshot_count": 0,
            "weak_snapshot_count": 0,
        }
        atomic_write_json(runtime_root / "cavers_stage1_eligibility.json", eligibility)

        source_rows = _source_files(data_root)
        download_fields = ("relative_path", "local_path", "size_bytes", "sha256", "status")
        atomic_write_csv(runtime_root / "download_manifest.csv", source_rows, download_fields)
        total_materialized = sum(row["size_bytes"] for row in source_rows)
        download_manifest = {
            "full_archive_download_count": 0,
            "large_lidar_or_point_cloud_download_count": 0,
            "materialized_bytes": total_materialized,
            "materialized_files": source_rows,
            "maximum_materialized_file_bytes": max(row["size_bytes"] for row in source_rows),
            "maximum_single_download_bytes": maximum_single_download_bytes,
            "range_response_bytes": evidence["total_range_response_bytes"],
            "resume_authenticated_sequence_count": len(
                evidence.get("resume_authenticated_sequence_ids", [])
            ),
            "resume_authenticated_sequence_ids": evidence.get(
                "resume_authenticated_sequence_ids", []
            ),
        }
        atomic_write_json(runtime_root / "download_manifest.json", download_manifest)

        primary = pair_selection["PRIMARY_CANDIDATE_MAP_QUERY_PAIR"]
        primary_text = (
            f"{primary['map_sequence_id']}→{primary['query_sequence_id']}" if primary else "无"
        )
        blocker_text = "、".join(eligibility["primary_hard_blockers"]) or "无"
        passed_diablo = [
            row["sequence_id"]
            for row in reference_rows
            if "diablo" in row["sequence_id"] and row["status"] == "PASS"
        ]
        passed_handheld = [
            row["sequence_id"]
            for row in reference_rows
            if "handheld" in row["sequence_id"] and row["status"] == "PASS"
        ]
        answers = [
            f"实际物化 {total_materialized / 1e6:.6f} MB（含论文、代码、GT/TF/时间索引与审计元数据）。",
            "否；没有下载任何完整 LiDAR、PCD 或 MCAP 大文件。",
            f"是；Zenodo exact record={ZENODO_RECORD_ID}/version={ZENODO_VERSION}/revision={ZENODO_REVISION}，GitHub commit={GITHUB_COMMIT}。",
            f"完整 OptiTrack 6DoF 的 DIABLO 序列：{','.join(passed_diablo) if passed_diablo else '无'}。",
            f"完整 OptiTrack 6DoF 的 HANDHELD 序列：{','.join(passed_handheld) if passed_handheld else '无'}；5/6 因无独立 GT 排除。",
            f"固定 world/map frame 判定：{'PASS' if shared_world else 'FAIL'}；同名且未归零的 map 已观察到，但官方未发布跨 recording 的共享 calibration/session 标识或不变性保证。",
            f"sequence-local reset：{'检测到 ' + ','.join(reset_sequences) if reset_sequences else '未检测到'}。",
            f"公开 rig→VLP16 外参：{'已找到' if extrinsic_status == 'PASS' else '未对全部候选找到'}。",
            "外参来自各 sequence 的 TF_STATIC/data.csv（官方 /tf_static export），精确来源和矩阵见 extrinsic audit。",
            f"时间链状态：{time_status}；共享记录时钟可审计，但无硬件级同步。",
            "hardware sync、sensor/processing delay、GT/LiDAR timestamp uncertainty 仍为 UNKNOWN。",
            f"R02={r02}。",
            f"R03={r03}。",
            f"R04={r04}。",
            f"R05={r05}。",
            f"R10={uncertainty_status}。",
            f"GT-only 最佳 pair：{primary_text}。",
            f"covered_duration_s={primary['total_covered_duration_s'] if primary else 'NOT_COMPUTABLE'}。",
            f"coverage_fraction={primary['coverage_fraction'] if primary else 'NOT_COMPUTABLE'}。",
            f"eligible 5s intervals={primary['eligible_nonoverlapping_5s_interval_count'] if primary else 'NOT_COMPUTABLE'}。",
            f"primary={primary_text}；reserve_1={'有' if pair_selection['RESERVE_PAIR_1'] else '无'}；reserve_2={'有' if pair_selection['RESERVE_PAIR_2'] else '无'}。",
            "ICP/registration 调用严格为 0。",
            "否；没有任何 >500,000,000 bytes 文件被完整下载或物化。",
            f"CAVERS_STAGE1_READY={str(ready).lower()}。",
            ("是；但只能在下一独立任务下载最小必要 VLP-16。" if ready else "否；Stage-1 硬门未通过，不进入 VLP-16 下载阶段。"),
            f"失败硬门：{blocker_text}；UNKNOWN uncertainty 没有被伪装成 0。",
        ]
        final_conclusion = (
            "CAVERS 已通过 metadata/GT/calibration/world-frame/GT-only-overlap/uncertainty-feasibility Stage-1，可以在下一独立任务中下载最小必要 VLP-16 数据，进行 geometry-only weak/rich blind selection。"
            if ready
            else "CAVERS Stage-1 未通过；失败硬门已在 cavers_stage1_eligibility.json 中冻结，不下载 VLP-16。"
        )
        summary = {
            "answers": answers,
            "cavers_stage1_ready": ready,
            "eligibility": eligibility,
            "final_conclusion": final_conclusion,
            "pair_selection": pair_selection,
        }
        atomic_write_json(runtime_root / "cavers_stage1_summary.json", summary)
        atomic_write_bytes(
            runtime_root / "cavers_stage1_summary.md", _summary_markdown(summary).encode("utf-8")
        )
        attestation = guard.attestation(runtime_root)
        attestation.update(
            {
                "estimated_transform_count": attestation.get(
                    "estimated_transform_count", attestation.get("estimated_transform_file_count", 0)
                ),
                "large_lidar_download_count": 0,
                "status": "PASS" if attestation["pass"] else "FAIL",
            }
        )
        atomic_write_json(runtime_root / "NO_ICP_ATTESTATION.json", attestation)

    manifest_exclusions = {"cavers_stage1_manifest.json", "SHA256SUMS"}
    payload_names = sorted(
        path.name
        for path in runtime_root.iterdir()
        if path.is_file() and path.name not in manifest_exclusions
    )
    manifest = {
        "data_evidence": source_rows,
        "eligibility_sha256": sha256_file(runtime_root / "cavers_stage1_eligibility.json"),
        "official_source_identity": {
            "github_commit": GITHUB_COMMIT,
            "zenodo_record_id": ZENODO_RECORD_ID,
            "zenodo_revision": ZENODO_REVISION,
            "zenodo_version": ZENODO_VERSION,
        },
        "payload": [
            {
                "path": name,
                "sha256": sha256_file(runtime_root / name),
                "size_bytes": (runtime_root / name).stat().st_size,
            }
            for name in payload_names
        ],
        "producer_commit": environment["git"]["commit"],
        "finalizing_commit": finalizing_environment["git"]["commit"],
        "schema_version": "cavers_stage1_manifest_v1",
    }
    manifest["manifest_payload_sha256"] = compact_sha256(manifest)
    atomic_write_json(runtime_root / "cavers_stage1_manifest.json", manifest)
    checksum_names = [*payload_names, "cavers_stage1_manifest.json"]
    atomic_write_bytes(runtime_root / "SHA256SUMS", _sha256sums_payload(runtime_root, checksum_names))
    return summary
