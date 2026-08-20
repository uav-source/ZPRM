#!/usr/bin/env python3
"""Independently verify the frozen FMB1 pre-registration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.preregistration_verify import (  # noqa: E402
    validate_manifest_payload,
)


DEFAULT_MANIFEST = REPOSITORY / "results/mid360_formal_batch1/fmb1_frozen_manifest.json"
DEFAULT_OUTPUT = REPOSITORY / "results/mid360_formal_batch1/fmb1_verification_report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_report_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write an idempotent report and never replace different bytes."""

    content = _encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise RuntimeError(
            f"refusing to overwrite a different verification report: {path}; "
            "select a new path with --output"
        )
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary report exists: {temporary}")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    except BaseException:
        # A failed write must not leave a file that could be mistaken for a
        # complete independent verifier report.
        if temporary.exists():
            temporary.unlink()
        raise


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("frozen manifest root must be a JSON object")
    return payload


def build_report(manifest_path: Path) -> tuple[dict[str, Any], int]:
    """Return a PASS/FAIL report and its intended process exit code."""

    manifest_text = str(manifest_path)
    manifest_sha256: str | None = None
    try:
        resolved = manifest_path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError(f"manifest is not a regular file: {resolved}")
        manifest_text = str(resolved)
        manifest_sha256 = _sha256_file(resolved)
        payload = _load_payload(resolved)
        verification = validate_manifest_payload(
            payload,
            REPOSITORY,
            verify_files=True,
        )
        report = {
            "schema": "mid360_fmb1_independent_verification_report_v1",
            "status": "PASS",
            "manifest_path": manifest_text,
            "manifest_file_sha256": manifest_sha256,
            "verify_files": True,
            "verification": verification,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_registration_trials": 0,
        }
        return report, 0
    except Exception as exc:
        report = {
            "schema": "mid360_fmb1_independent_verification_report_v1",
            "status": "FAIL",
            "manifest_path": manifest_text,
            "manifest_file_sha256": manifest_sha256,
            "verify_files": True,
            "verification": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "actual_registration_trials": 0,
        }
        return report, 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = REPOSITORY / manifest_path
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = REPOSITORY / output_path

    report, exit_code = build_report(manifest_path)
    try:
        _write_report_once(output_path, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "report_written": False,
                    "output": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": report["status"],
                "report_written": True,
                "output": str(output_path),
                "manifest_file_sha256": report["manifest_file_sha256"],
                "error": report.get("error"),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
