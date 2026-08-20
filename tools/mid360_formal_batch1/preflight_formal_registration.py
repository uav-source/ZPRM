#!/usr/bin/env python3
"""Read-only, fail-closed preflight for the real FMB1 backend boundary.

The command prints evidence only.  It never writes a trial matrix, lock,
fingerprint, authorization token, or backend result and never calls a backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.mid360_formal_batch1.registration_firewall import (  # noqa: E402
    QUALIFICATION_RUNTIME_RELATIVE,
    REAL_RUNTIME_RELATIVE,
    REAL_RESULTS_RELATIVE,
    REAL_SCOPE,
    preflight_real_batch,
    write_no_icp_attestation_tonight,
    write_preflight_reports,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", "--repository", dest="repository_root", type=Path,
        default=REPOSITORY,
    )
    parser.add_argument(
        "--results-root", "--results-dir", dest="results_root", type=Path
    )
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--qualification-runtime-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="explicitly attest that this invocation is evidence-only",
    )
    args = parser.parse_args(argv)

    repository = args.repository_root.resolve(strict=True)
    results_dir = args.results_root or repository / REAL_RESULTS_RELATIVE
    if not (results_dir / "fmb1_pre_registration_readiness.json").is_file():
        nested = results_dir / "mid360_formal_batch1"
        if (nested / "fmb1_pre_registration_readiness.json").is_file():
            results_dir = nested
    runtime_dir = args.runtime_dir or repository / REAL_RUNTIME_RELATIVE
    qualification_runtime_dir = (
        args.qualification_runtime_dir
        or repository / QUALIFICATION_RUNTIME_RELATIVE
    )
    output_dir = args.output_dir or results_dir / "prebackend_qualification_v1"
    if not args.no_backend:
        refusal = {
            "schema": "mid360_fmb1_formal_registration_preflight_cli_refusal_v1",
            "status": "BLOCKED",
            "pass": False,
            "CURRENT_REAL_BATCH_BLOCKED": True,
            "CURRENT_BLOCK_REASON": "EXPLICIT_NO_BACKEND_ACK_REQUIRED",
            "NO_BACKEND_MODE": False,
            "FORMAL_RUN_MATRIX_ISSUED": False,
            "FORMAL_LOCK_ISSUED": False,
            "FORMAL_REGISTRATION_AUTHORIZED": False,
            "FORMAL_ICP_UNLOCKED": False,
            "backend_invoked": False,
            "evidence_written": False,
        }
        print(json.dumps(refusal, indent=2, sort_keys=True, ensure_ascii=False))
        return 2
    os.environ.setdefault("ZPRM_FMB1_NO_FORMAL_REGISTRATION", "1")
    os.environ.setdefault("NO_FORMAL_REGISTRATION", "true")
    os.environ.setdefault("FMB1_EXECUTION_SCOPE", REAL_SCOPE)

    report = preflight_real_batch(
        repository,
        results_dir=results_dir,
        runtime_dir=runtime_dir,
        qualification_runtime_dir=qualification_runtime_dir,
    )
    report["NO_BACKEND_MODE"] = True
    report["no_backend_cli_flag_supplied"] = bool(args.no_backend)
    written = write_preflight_reports(report, output_dir)
    report["written_reports"] = written
    try:
        report["tonight_no_icp_attestation"] = write_no_icp_attestation_tonight(
            report, output_dir
        )
    except Exception as exc:
        report["tonight_no_icp_attestation"] = {
            "status": "NOT_ISSUED",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("pass") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
