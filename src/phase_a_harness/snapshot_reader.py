"""Strict reader for copied Stage-0 snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import canonical_json_sha256, file_sha256


EXPECTED_FILES = frozenset(
    {
        "metadata.json",
        "source_points.npy",
        "target_points.npy",
        "reference_pose.npy",
        "source_parent_target_indices.npy",
    }
)


def snapshot_directory(cache_root: str | Path, snapshot_id: str) -> Path:
    root = Path(cache_root).resolve()
    candidate = (root / snapshot_id).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("snapshot path escaped cache root")
    return candidate


def read_snapshot(cache_root: str | Path, snapshot_id: str, *, arrays: bool = True) -> dict[str, Any]:
    directory = snapshot_directory(cache_root, snapshot_id)
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != EXPECTED_FILES:
        raise ValueError(f"snapshot file inventory mismatch: {snapshot_id}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    payload = dict(metadata)
    stored = payload.pop("metadata_payload_sha256", None)
    if stored != canonical_json_sha256(payload) or metadata.get("snapshot_id") != snapshot_id:
        raise ValueError(f"snapshot metadata mismatch: {snapshot_id}")
    for name, digest in metadata["array_file_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"snapshot array SHA mismatch: {snapshot_id}/{name}")
    result: dict[str, Any] = {"directory": directory, "metadata": metadata}
    if not arrays:
        return result
    source = np.load(directory / "source_points.npy", allow_pickle=False)
    target = np.load(directory / "target_points.npy", allow_pickle=False)
    reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    parent = np.load(directory / "source_parent_target_indices.npy", allow_pickle=False)
    if source.dtype != np.dtype("<f4") or target.dtype != np.dtype("<f4"):
        raise ValueError(f"snapshot point dtype mismatch: {snapshot_id}")
    if reference.dtype != np.dtype("<f8") or parent.dtype.kind not in "iu":
        raise ValueError(f"snapshot reference/index dtype mismatch: {snapshot_id}")
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"snapshot point shape mismatch: {snapshot_id}")
    if reference.shape != (4, 4) or parent.shape != (source.shape[0],):
        raise ValueError(f"snapshot reference/index shape mismatch: {snapshot_id}")
    if not all(item.flags.c_contiguous for item in (source, target, reference, parent)):
        raise ValueError(f"snapshot arrays are not C-contiguous: {snapshot_id}")
    if not all(np.all(np.isfinite(item)) for item in (source, target, reference)):
        raise ValueError(f"snapshot contains non-finite data: {snapshot_id}")
    result.update(source=source, target=target, reference=reference, parent_indices=parent)
    return result


__all__ = ["EXPECTED_FILES", "read_snapshot", "snapshot_directory"]

