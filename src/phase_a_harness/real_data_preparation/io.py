"""Canonical and fail-closed filesystem primitives for preparation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class PreparationIOError(RuntimeError):
    """Raised when a path or immutable artifact is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
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


def compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_root(path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PreparationIOError(f"path is not absolute: {candidate}")
    if candidate.exists() and candidate.is_symlink():
        raise PreparationIOError(f"root is a symlink: {candidate}")
    resolved = candidate.resolve(strict=must_exist)
    if resolved != candidate:
        raise PreparationIOError(f"path is not canonical: {candidate} != {resolved}")
    return resolved


def assert_within(root: Path, path: Path) -> None:
    root = root.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    if parent != root and root not in parent.parents:
        raise PreparationIOError(f"path escapes root: {path}")
    cursor = parent
    while cursor != root:
        if cursor.is_symlink():
            raise PreparationIOError(f"symlink component is forbidden: {cursor}")
        cursor = cursor.parent


def atomic_write_bytes(path: str | Path, payload: bytes, *, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        existing = destination.read_bytes()
        if existing != payload:
            raise PreparationIOError(f"immutable artifact differs: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    overwrite: bool = False,
) -> None:
    atomic_write_bytes(path, csv_bytes(rows, fields), overwrite=overwrite)
