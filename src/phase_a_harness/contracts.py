"""Single-manifest and immutable scientific-contract helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
SOURCE_COMMIT = "89f46dda68e9ff5c71f078f6d13fc9050d58f0f5"
SOURCE_BUNDLE_SHA256 = "5f76ce49910bea6bf7c85c8e95bc2610948b7e22491e1209818a08f80bc591fc"
OPEN3D_PLAN_BACKEND = "open3d_point_to_plane"
PCL_PLAN_BACKEND = "pcl_iterative_closest_point_with_normals"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_manifest(path: str | Path, *, require_authorized: bool = False) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).resolve()
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if type(value) is not dict or value.get("manifest_version") != "1":
        raise ValueError("frozen experiment manifest identity mismatch")
    stored = value.get("manifest_payload_sha256")
    payload = {key: item for key, item in value.items() if key != "manifest_payload_sha256"}
    if stored != canonical_json_sha256(payload):
        raise ValueError("frozen experiment manifest payload SHA mismatch")
    if require_authorized and value.get("formal_execution_authorized") is not True:
        raise PermissionError("formal execution is not authorized")
    return candidate, value


def manifest_root(path: Path) -> Path:
    if path.parent.name != "frozen_assets":
        raise ValueError("manifest must reside in frozen_assets")
    return path.parent.parent.resolve()


__all__ = [
    "OPEN3D_PLAN_BACKEND",
    "PCL_PLAN_BACKEND",
    "SOURCE_BUNDLE_SHA256",
    "SOURCE_COMMIT",
    "SOURCE_REPOSITORY",
    "canonical_json_sha256",
    "file_sha256",
    "load_manifest",
    "manifest_root",
    "write_json",
]

