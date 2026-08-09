#!/usr/bin/env python3
"""Run one external, seed-free runtime lifecycle fixture invocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import threading
from pathlib import Path
from typing import Any, Sequence


FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
SOURCE_ACCESS_EVENT_SCHEMA = "runtime_lifecycle_source_access_event_v1"


def _overlaps(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _search_path_reaches_source(entry: str, *, source: Path) -> bool:
    try:
        candidate = (Path.cwd() if not entry else Path(entry)).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PermissionError(f"invalid Python search path entry: {entry!r}") from error
    return _overlaps(candidate, source)


def _assert_isolation() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("runtime lifecycle fixture requires PYTHONNOUSERSITE=1")
    if os.environ.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError("runtime lifecycle fixture requires the frozen MAMBA_ROOT_PREFIX")
    source = SOURCE_REPOSITORY.resolve()
    entries = list(sys.path) + [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    for entry in entries:
        if _search_path_reaches_source(entry, source=source):
            raise PermissionError(
                "Python search path is the source repository, a descendant, "
                "or an ancestor containing it"
            )


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"source-access log path contains symlink: {current}")


def _prepare_source_access_log(path: Path, *, repository: Path) -> Path:
    """Validate and create the external append-only evidence file."""

    if not path.is_absolute():
        raise PermissionError("--source-access-log must be an absolute external path")
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical != path:
        raise PermissionError("--source-access-log must already be canonical")
    try:
        os.lstat(lexical)
    except FileNotFoundError:
        pass
    else:
        # Refuse regular files, links, FIFOs, devices, sockets, and directories
        # before any open.  O_EXCL below closes the lstat/open race as well.
        raise FileExistsError(
            f"--source-access-log must not already exist: {lexical}"
        )
    _assert_no_symlink_components(lexical.parent)
    candidate = lexical.resolve(strict=False)
    source = SOURCE_REPOSITORY.resolve()
    if _overlaps(candidate, repository.resolve()) or _overlaps(candidate, source):
        raise PermissionError(
            "--source-access-log must be outside both repositories"
        )
    if not candidate.parent.is_dir():
        raise PermissionError("--source-access-log parent directory must already exist")
    _assert_no_symlink_components(candidate.parent)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags, 0o600)
    except OSError as error:
        raise PermissionError(
            f"cannot prepare external source-access log: {candidate}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PermissionError("--source-access-log must be a regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        candidate.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return candidate


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("zero-byte source-access evidence write")
        remaining = remaining[written:]


class _EarlySourceAccessMonitor:
    """Audit source opens before importing any harness runtime module."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.count = 0
        self.paths: list[str] = []
        self._installed = False
        self._lock = threading.Lock()

    def install(self) -> None:
        if self._installed:
            return
        source = SOURCE_REPOSITORY.resolve()
        log_path = self.log_path

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            raw = os.fsdecode(args[0])
            try:
                candidate = Path(raw).resolve()
            except (OSError, RuntimeError, TypeError, ValueError):
                return
            if candidate != source and source not in candidate.parents:
                return
            with self._lock:
                self.count += 1
                self.paths.append(str(candidate))
                evidence = {
                    "event": "open",
                    "event_schema": SOURCE_ACCESS_EVENT_SCHEMA,
                    "path": str(candidate),
                    "pid": os.getpid(),
                    "sequence": self.count,
                    "source_repository": str(source),
                }
                payload = (
                    json.dumps(
                        evidence,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                flags = (
                    os.O_WRONLY
                    | os.O_APPEND
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(log_path, flags)
                    try:
                        _write_all(descriptor, payload)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                except OSError as error:
                    raise PermissionError(
                        "source repository read forbidden and external audit logging failed"
                    ) from error
            raise PermissionError(
                f"source repository runtime read forbidden: {candidate}"
            )

        sys.addaudithook(audit)
        self._installed = True


def _source_access_report(
    *,
    early_monitor: _EarlySourceAccessMonitor,
    source_monitor: Any,
    import_paths: Sequence[str],
) -> dict[str, Any]:
    early_paths = list(early_monitor.paths)
    secondary_paths = list(getattr(source_monitor, "paths", []))
    paths = sorted(set(early_paths + secondary_paths))
    early_count = int(early_monitor.count)
    secondary_count = int(getattr(source_monitor, "count", 0))
    return {
        "source_access_log_path": str(early_monitor.log_path),
        "source_repository_early_file_read_count": early_count,
        "source_repository_early_file_read_paths": early_paths,
        "source_repository_runtime_file_read_count": early_count + secondary_count,
        "source_repository_runtime_file_read_paths": paths,
        "source_repository_runtime_import_count": len(import_paths),
        "source_repository_runtime_import_paths": list(import_paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the seed-free external runtime lifecycle fixture"
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--source-access-log", type=Path, required=True)
    parser.add_argument("--qualification-delay-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_isolation()
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    source_access_log = _prepare_source_access_log(
        args.source_access_log, repository=repository
    )
    early_source_monitor = _EarlySourceAccessMonitor(source_access_log)
    early_source_monitor.install()
    sys.path.insert(0, str(repository / "src"))
    # Install the established in-process monitor before importing any other
    # phase_a_harness runtime component.  The earlier stdlib-only hook above
    # also covers the import of runner.py and all of its transitive imports.
    from phase_a_harness.runner import SourceAccessMonitor

    source_monitor = SourceAccessMonitor()
    source_monitor.install()
    from phase_a_harness.runtime_git_gate import verify_runtime_git_gate
    from phase_a_harness.runtime_lifecycle_fixture import run_fixture_lifecycle
    from phase_a_harness.runtime_lifecycle_io import (
        atomic_create_canonical_json,
        canonical_json_sha256,
    )
    from phase_a_harness.runtime_path_policy import (
        POLICY_SCHEMA,
        qualify_runtime_paths,
    )
    from phase_a_harness.asset_verifier import source_runtime_import_paths

    qualification = qualify_runtime_paths(
        args.run_id,
        runtime_root=args.runtime_root,
        repository_root=repository,
        run_kind="fixture",
        resume=args.resume,
    )
    layout = qualification.layout
    policy_binding = {
        "schema_version": POLICY_SCHEMA,
        "layout": layout.as_dict(),
    }
    policy_sha = canonical_json_sha256(policy_binding)
    gate_reports: list[dict[str, Any]] = []
    gate_directory = layout.temporary_inventory / "git_gates"
    next_gate_sequence = 1
    if gate_directory.exists():
        names = sorted(path.name for path in gate_directory.iterdir())
        numbers = []
        for name in names:
            match = re.fullmatch(r"([0-9]{4})_[A-Za-z0-9_-]+\.json", name)
            if match is None:
                raise ValueError(f"unexpected Git-gate evidence file: {name}")
            numbers.append(int(match.group(1)))
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("Git-gate evidence sequence is not contiguous")
        next_gate_sequence = len(numbers) + 1

    def gate(checkpoint: str) -> dict[str, Any]:
        nonlocal next_gate_sequence
        report = verify_runtime_git_gate(
            repository,
            args.expected_commit,
            args.expected_branch,
            args.expected_tag,
            checkpoint=checkpoint,
        )
        gate_reports.append(report)
        if layout.run_root.exists():
            gate_directory.mkdir(parents=True, exist_ok=True)
            while gate_reports:
                current = gate_reports.pop(0)
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", current["checkpoint"])
                atomic_create_canonical_json(
                    gate_directory / f"{next_gate_sequence:04d}_{safe}.json",
                    current,
                )
                next_gate_sequence += 1
        return report

    result = run_fixture_lifecycle(
        repository=repository,
        layout=layout,
        run_id=args.run_id,
        invocation_id=args.invocation_id,
        workers=args.workers,
        resume=args.resume,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tag=args.expected_tag,
        runtime_path_policy_sha256=policy_sha,
        qualification_delay_seconds=args.qualification_delay_seconds,
        git_gate=gate,
    )
    if gate_reports:
        gate("FINAL_GIT_GATE_FLUSH")
    import_paths = source_runtime_import_paths()
    source_access = _source_access_report(
        early_monitor=early_source_monitor,
        source_monitor=source_monitor,
        import_paths=import_paths,
    )
    result = {
        **result,
        "runtime_path_audit": dict(qualification.audit),
        "runtime_path_policy_binding": policy_binding,
        "runtime_path_policy_sha256": policy_sha,
        "git_gate_evidence_count": next_gate_sequence - 1,
        **source_access,
    }
    if (
        result["source_repository_runtime_file_read_count"] != 0
        or result["source_repository_runtime_import_count"] != 0
    ):
        raise PermissionError("source repository runtime isolation failed")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
