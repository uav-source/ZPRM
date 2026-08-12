"""Production-scale deterministic target-map reduction from authenticated replay.

The existing Python voxel builder remains the executable specification for
small fixtures.  Real Boreas contains billions of raw points, so this module
drives a source-pinned, single-threaded C++ reducer whose arithmetic and output
ordering match that specification without retaining Python objects per voxel.

This module has no downloader and no registration backend entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .io import atomic_write_bytes, sha256_file
from .streaming_target_map import VoxelRule, canonical_array_sha256


SOURCE_RELATIVE = Path("boreas_stage2_voxel_reduce.cpp")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_COUNT_STRUCT = struct.Struct("<Q")
POINT_BYTES = 24
CAPACITY_LAYOUT_SCHEMA = "zprm-boreas-stage2-reducer-capacity-layout-v1"
TARGET_REPLAY_VERIFICATION_SCHEMA = (
    "zprm.boreas.stage2.target_replay_verification.v1"
)


class ExternalVoxelReductionError(RuntimeError):
    """An authenticated replay or external reduction invariant failed."""


class DiskGate(Protocol):
    def before_materialization(
        self, projected_write_bytes: int, *, artifact_id: str
    ) -> Any: ...


@dataclass(frozen=True)
class AuthenticatedReplayRange:
    byte_offset: int
    point_count: int
    transformed_xyz_sha256: str

    @classmethod
    def from_value(
        cls, value: "AuthenticatedReplayRange | Mapping[str, Any]"
    ) -> "AuthenticatedReplayRange":
        if isinstance(value, cls):
            result = value
        else:
            result = cls(
                byte_offset=int(value["byte_offset"]),
                point_count=int(value["point_count"]),
                transformed_xyz_sha256=str(value["transformed_xyz_sha256"]),
            )
        if result.byte_offset < 0 or result.point_count < 0:
            raise ExternalVoxelReductionError("replay range offset/count is invalid")
        if SHA256_RE.fullmatch(result.transformed_xyz_sha256) is None:
            raise ExternalVoxelReductionError("replay range SHA-256 is invalid")
        return result


@dataclass(frozen=True)
class ExternalTargetMapResult:
    target_points_path: Path
    target_points_file_sha256: str
    target_array_sha256: str
    point_count: int
    voxel_rule_sha256: str
    reducer_binary_sha256: str
    range_descriptor_sha256: str


@dataclass(frozen=True)
class ExternalTargetReplayVerificationResult:
    """Exact replay-to-target comparison performed without a second target copy."""

    target_points_path: Path
    target_points_file_sha256: str
    point_count: int
    compared_npy_bytes: int
    reducer_binary_sha256: str
    range_descriptor_sha256: str
    estimated_peak_memory_bytes: int
    live_available_memory_bytes: int
    verification_status: str


def _canonical_path(path: str | Path, *, must_exist: bool, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink():
        raise ExternalVoxelReductionError(f"{label} cannot be a symlink")
    if must_exist:
        result = candidate.resolve(strict=True)
        if not result.is_file() or result.is_symlink():
            raise ExternalVoxelReductionError(f"{label} must be a regular file")
        return result
    parent = candidate.parent.resolve(strict=True)
    result = parent / candidate.name
    if result != candidate or result.exists():
        raise ExternalVoxelReductionError(
            f"{label} must be a new canonical path in an existing directory"
        )
    return result


def write_range_descriptor(
    path: str | Path,
    ranges: Sequence[AuthenticatedReplayRange | Mapping[str, Any]],
) -> tuple[Path, str]:
    """Write the closed, canonical range view consumed by the C++ reducer."""

    output = Path(path)
    rows = [AuthenticatedReplayRange.from_value(value) for value in ranges]
    if not rows:
        raise ExternalVoxelReductionError("at least one replay range is required")
    offsets = [row.byte_offset for row in rows]
    if offsets != sorted(set(offsets)):
        raise ExternalVoxelReductionError(
            "replay range offsets must be unique and strictly increasing"
        )
    for previous, current in zip(rows, rows[1:]):
        if previous.byte_offset + previous.point_count * POINT_BYTES > current.byte_offset:
            raise ExternalVoxelReductionError("replay active ranges overlap")
    payload = "byte_offset\tpoint_count\tsha256\n" + "".join(
        f"{row.byte_offset}\t{row.point_count}\t{row.transformed_xyz_sha256}\n"
        for row in rows
    )
    atomic_write_bytes(output, payload.encode("ascii"))
    return output.resolve(strict=True), sha256_file(output)


def compile_external_voxel_reducer(
    *, source: str | Path, output: str | Path, compiler: str = "g++"
) -> dict[str, Any]:
    """Compile the repository source with an exact, recorded command."""

    source_path = _canonical_path(source, must_exist=True, label="reducer source")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and (output_path.is_symlink() or not output_path.is_file()):
        raise ExternalVoxelReductionError("reducer output path is unsafe")
    compiler_path_text = shutil.which(compiler)
    if compiler_path_text is None:
        raise ExternalVoxelReductionError(f"C++ compiler unavailable: {compiler}")
    compiler_path = Path(compiler_path_text).resolve(strict=True)
    temporary = output_path.parent / f".{output_path.name}.compile-partial-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise ExternalVoxelReductionError("reducer compile temporary already exists")
    command = [
        str(compiler_path),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        str(source_path),
        "-lcrypto",
        "-o",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise ExternalVoxelReductionError(
                "reducer compilation failed: " + completed.stderr[-4000:]
            )
        os.chmod(temporary, 0o500)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, output_path)
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    version = subprocess.run(
        [str(compiler_path), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "binary_path": str(output_path.resolve(strict=True)),
        "binary_sha256": sha256_file(output_path),
        "command": command,
        "compiler_version_first_line": version,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
    }


def _read_exact(stream: Any, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        value = stream.read(remaining)
        if not value:
            raise ExternalVoxelReductionError("external reducer stdout is truncated")
        chunks.append(value)
        remaining -= len(value)
    return b"".join(chunks)


def describe_external_voxel_capacity(
    *,
    reducer_binary: str | Path,
    reducer_binary_sha256: str,
    max_voxels: int,
) -> dict[str, Any]:
    """Read the compiled reducer's ABI/layout-derived capacity envelope."""

    binary = _canonical_path(reducer_binary, must_exist=True, label="reducer binary")
    if (
        SHA256_RE.fullmatch(str(reducer_binary_sha256)) is None
        or sha256_file(binary) != str(reducer_binary_sha256)
    ):
        raise ExternalVoxelReductionError("reducer binary SHA-256 differs")
    if isinstance(max_voxels, bool) or int(max_voxels) <= 0:
        raise ExternalVoxelReductionError("max_voxels must be a positive integer")
    completed = subprocess.run(
        [
            str(binary),
            "--print-capacity-layout",
            "--max-voxels",
            str(int(max_voxels)),
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ExternalVoxelReductionError(
            "reducer capacity description failed: " + completed.stderr[-4000:]
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalVoxelReductionError(
            "reducer capacity description is not JSON"
        ) from exc
    if not isinstance(value, dict) or value.get("schema") != CAPACITY_LAYOUT_SCHEMA:
        raise ExternalVoxelReductionError("reducer capacity schema differs")
    expected_line = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    if completed.stdout != expected_line:
        raise ExternalVoxelReductionError(
            "reducer capacity description is not canonical JSON"
        )
    return value


def _live_available_memory_bytes() -> int:
    try:
        rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        value = next(row for row in rows if row.startswith("MemAvailable:"))
        fields = value.split()
        if len(fields) != 3 or fields[2] != "kB":
            raise ValueError("MemAvailable has an unexpected schema")
        result = int(fields[1]) * 1024
    except (OSError, StopIteration, UnicodeError, ValueError) as exc:
        raise ExternalVoxelReductionError(
            "cannot establish live MemAvailable for replay verification"
        ) from exc
    if result <= 0:
        raise ExternalVoxelReductionError("live MemAvailable is not positive")
    return result


def verify_external_target_map_replay(
    *,
    replay_path: str | Path,
    range_descriptor_path: str | Path,
    range_descriptor_sha256: str,
    reducer_binary: str | Path,
    reducer_binary_sha256: str,
    voxel_rule: VoxelRule,
    target_points_path: str | Path,
    target_points_file_sha256: str,
    target_point_count: int,
    max_voxels: int,
    memory_safety_margin_bytes: int,
    available_memory_provider: Any = _live_available_memory_bytes,
) -> ExternalTargetReplayVerificationResult:
    """Replay authenticated ranges and compare exact target NPY bytes in place.

    The C++ process retains only its bounded voxel table, sorted-index vector,
    and small I/O buffers.  The target is opened read-only with ``O_NOFOLLOW``
    and compared while locked; no target-sized temporary is materialized.
    """

    replay = _canonical_path(replay_path, must_exist=True, label="replay array")
    descriptor = _canonical_path(
        range_descriptor_path, must_exist=True, label="range descriptor"
    )
    binary = _canonical_path(reducer_binary, must_exist=True, label="reducer binary")
    target = _canonical_path(target_points_path, must_exist=True, label="target points")
    for value, actual, label in (
        (range_descriptor_sha256, sha256_file(descriptor), "range descriptor"),
        (reducer_binary_sha256, sha256_file(binary), "reducer binary"),
        (target_points_file_sha256, sha256_file(target), "target points"),
    ):
        if SHA256_RE.fullmatch(str(value)) is None or str(value) != actual:
            raise ExternalVoxelReductionError(f"{label} SHA-256 differs")
    if not isinstance(voxel_rule, VoxelRule):
        raise ExternalVoxelReductionError("voxel_rule must be a frozen VoxelRule")
    for value, label in (
        (target_point_count, "target_point_count"),
        (max_voxels, "max_voxels"),
    ):
        if isinstance(value, bool) or int(value) <= 0:
            raise ExternalVoxelReductionError(f"{label} must be a positive integer")
    if int(target_point_count) > int(max_voxels):
        raise ExternalVoxelReductionError("target point count exceeds max_voxels")
    if isinstance(memory_safety_margin_bytes, bool) or int(
        memory_safety_margin_bytes
    ) < 0:
        raise ExternalVoxelReductionError(
            "memory_safety_margin_bytes must be a nonnegative integer"
        )
    try:
        target_view = np.load(target, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ExternalVoxelReductionError("target points NPY is invalid") from exc
    if (
        not isinstance(target_view, np.memmap)
        or target_view.dtype != np.dtype("<f8")
        or target_view.shape != (int(target_point_count), 3)
        or not target_view.flags.c_contiguous
    ):
        raise ExternalVoxelReductionError("target points NPY contract differs")
    del target_view

    layout = describe_external_voxel_capacity(
        reducer_binary=binary,
        reducer_binary_sha256=str(reducer_binary_sha256),
        max_voxels=int(max_voxels),
    )
    estimated = int(layout["total_peak_upper_bound_bytes"])
    try:
        available = int(available_memory_provider())
    except Exception as exc:
        raise ExternalVoxelReductionError(
            "available-memory provider failed"
        ) from exc
    required = estimated + int(memory_safety_margin_bytes)
    if available < required:
        raise ExternalVoxelReductionError(
            "insufficient live memory for bounded target replay verification: "
            f"available={available}, required={required}"
        )

    origin = voxel_rule.origin_xyz_m
    command = [
        str(binary),
        "--replay",
        str(replay),
        "--descriptor",
        str(descriptor),
        "--descriptor-sha256",
        str(range_descriptor_sha256),
        "--voxel-size-m",
        repr(voxel_rule.voxel_size_m),
        "--origin-x-m",
        repr(origin[0]),
        "--origin-y-m",
        repr(origin[1]),
        "--origin-z-m",
        repr(origin[2]),
        "--max-voxels",
        str(int(max_voxels)),
        "--verify-target-npy",
        str(target),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ExternalVoxelReductionError(
            "target replay verification failed: " + completed.stderr[-4000:]
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalVoxelReductionError(
            "target replay verification output is not JSON"
        ) from exc
    expected_fields = {
        "compared_npy_bytes",
        "descriptor_sha256",
        "replay_range_count",
        "schema",
        "status",
        "target_npy_sha256",
        "target_point_count",
    }
    if not isinstance(report, dict) or set(report) != expected_fields:
        raise ExternalVoxelReductionError(
            "target replay verification report schema differs"
        )
    expected_line = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    if completed.stdout != expected_line:
        raise ExternalVoxelReductionError(
            "target replay verification report is not canonical JSON"
        )
    expected = {
        "compared_npy_bytes": target.stat().st_size,
        "descriptor_sha256": str(range_descriptor_sha256),
        "schema": TARGET_REPLAY_VERIFICATION_SCHEMA,
        "status": "PASS_EXACT_TARGET_NPY_REPLAY",
        "target_npy_sha256": str(target_points_file_sha256),
        "target_point_count": int(target_point_count),
    }
    for field, value in expected.items():
        if report[field] != value:
            raise ExternalVoxelReductionError(
                f"target replay verification {field} differs"
            )
    try:
        descriptor_rows = descriptor.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExternalVoxelReductionError(
            "cannot count authenticated replay descriptor rows"
        ) from exc
    expected_range_count = len(descriptor_rows) - 1
    if (
        expected_range_count <= 0
        or isinstance(report["replay_range_count"], bool)
        or report["replay_range_count"] != expected_range_count
    ):
        raise ExternalVoxelReductionError(
            "target replay verification range count is invalid"
        )
    return ExternalTargetReplayVerificationResult(
        target_points_path=target,
        target_points_file_sha256=str(target_points_file_sha256),
        point_count=int(target_point_count),
        compared_npy_bytes=int(report["compared_npy_bytes"]),
        reducer_binary_sha256=str(reducer_binary_sha256),
        range_descriptor_sha256=str(range_descriptor_sha256),
        estimated_peak_memory_bytes=estimated,
        live_available_memory_bytes=available,
        verification_status=str(report["status"]),
    )


def build_external_target_map(
    *,
    replay_path: str | Path,
    range_descriptor_path: str | Path,
    range_descriptor_sha256: str,
    reducer_binary: str | Path,
    reducer_binary_sha256: str,
    voxel_rule: VoxelRule,
    target_points_path: str | Path,
    max_voxels: int,
    disk_gate: DiskGate,
    io_chunk_rows: int = 131_072,
) -> ExternalTargetMapResult:
    """Build one canonical target NPY while holding no Python per-voxel state."""

    replay = _canonical_path(replay_path, must_exist=True, label="replay array")
    descriptor = _canonical_path(
        range_descriptor_path, must_exist=True, label="range descriptor"
    )
    binary = _canonical_path(reducer_binary, must_exist=True, label="reducer binary")
    output = _canonical_path(
        target_points_path, must_exist=False, label="target points output"
    )
    for value, actual, label in (
        (range_descriptor_sha256, sha256_file(descriptor), "range descriptor"),
        (reducer_binary_sha256, sha256_file(binary), "reducer binary"),
    ):
        if SHA256_RE.fullmatch(str(value)) is None or str(value) != actual:
            raise ExternalVoxelReductionError(f"{label} SHA-256 differs")
    if not isinstance(voxel_rule, VoxelRule):
        raise ExternalVoxelReductionError("voxel_rule must be a frozen VoxelRule")
    if isinstance(max_voxels, bool) or int(max_voxels) <= 0:
        raise ExternalVoxelReductionError("max_voxels must be a positive integer")
    if isinstance(io_chunk_rows, bool) or int(io_chunk_rows) <= 0:
        raise ExternalVoxelReductionError("io_chunk_rows must be positive")

    origin = voxel_rule.origin_xyz_m
    command = [
        str(binary),
        "--replay",
        str(replay),
        "--descriptor",
        str(descriptor),
        "--descriptor-sha256",
        str(range_descriptor_sha256),
        "--voxel-size-m",
        repr(voxel_rule.voxel_size_m),
        "--origin-x-m",
        repr(origin[0]),
        "--origin-y-m",
        repr(origin[1]),
        "--origin-z-m",
        repr(origin[2]),
        "--max-voxels",
        str(int(max_voxels)),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    temporary = output.parent / f".{output.name}.materialize-partial-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        process.kill()
        process.wait()
        raise ExternalVoxelReductionError("target materialization temporary exists")
    try:
        try:
            count_payload = _read_exact(process.stdout, 8)
        except ExternalVoxelReductionError as error:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            returncode = process.wait()
            raise ExternalVoxelReductionError(
                f"external reducer failed ({returncode}): {stderr[-4000:]}"
            ) from error
        count = OUTPUT_COUNT_STRUCT.unpack(count_payload)[0]
        if count == 0 or count > int(max_voxels):
            raise ExternalVoxelReductionError("external reducer emitted an invalid voxel count")
        projected = int(count) * POINT_BYTES + 4096
        disk_gate.before_materialization(
            projected,
            artifact_id="TARGET_MAP_CANONICAL_NPY:target_maps/<pending-sha256>/target_points.npy",
        )
        array = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.dtype("<f8"),
            shape=(int(count), 3),
            fortran_order=False,
            version=(1, 0),
        )
        for start in range(0, int(count), int(io_chunk_rows)):
            stop = min(int(count), start + int(io_chunk_rows))
            payload = _read_exact(process.stdout, (stop - start) * POINT_BYTES)
            array[start:stop] = np.frombuffer(payload, dtype="<f8").reshape((-1, 3))
        extra = process.stdout.read(1)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        if returncode != 0:
            raise ExternalVoxelReductionError(
                f"external reducer failed ({returncode}): {stderr[-4000:]}"
            )
        if extra:
            raise ExternalVoxelReductionError("external reducer emitted trailing stdout")
        array.flush()
        del array
        descriptor_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)
        check = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if (
            check.dtype != np.dtype("<f8")
            or check.shape != (int(count), 3)
            or not check.flags.c_contiguous
            or not np.all(np.isfinite(check))
        ):
            raise ExternalVoxelReductionError("materialized target NPY is noncanonical")
        array_sha = canonical_array_sha256(check)
        del check
        file_sha = sha256_file(temporary)
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return ExternalTargetMapResult(
        target_points_path=output,
        target_points_file_sha256=file_sha,
        target_array_sha256=array_sha,
        point_count=int(count),
        voxel_rule_sha256=voxel_rule.contract_sha256,
        reducer_binary_sha256=str(reducer_binary_sha256),
        range_descriptor_sha256=str(range_descriptor_sha256),
    )


def default_reducer_source() -> Path:
    return Path(__file__).resolve().parent / SOURCE_RELATIVE


__all__ = [
    "AuthenticatedReplayRange",
    "ExternalTargetMapResult",
    "ExternalTargetReplayVerificationResult",
    "ExternalVoxelReductionError",
    "CAPACITY_LAYOUT_SCHEMA",
    "TARGET_REPLAY_VERIFICATION_SCHEMA",
    "build_external_target_map",
    "compile_external_voxel_reducer",
    "describe_external_voxel_capacity",
    "default_reducer_source",
    "verify_external_target_map_replay",
    "write_range_descriptor",
]
