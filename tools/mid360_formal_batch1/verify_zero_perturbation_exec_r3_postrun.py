#!/usr/bin/env python3
"""CLI for the frozen independent FMB1 Exec-R3 post-run verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.mid360_formal_batch1.postrun_verification.independent_postrun_verifier_v1 import (  # noqa: E402
    IndependentPostrunVerificationError,
    fixture_only_self_test,
    verify_frozen_real_results,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fixture-only", action="store_true")
    result.add_argument("--repository-root", type=Path)
    result.add_argument("--execution-results-root", type=Path)
    result.add_argument("--r3-lock-dir", type=Path)
    result.add_argument("--raw-execution-commit")
    result.add_argument("--raw-execution-tag")
    result.add_argument("--verifier-code-commit")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--confirm-read-frozen-real-results", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args=parser().parse_args(argv)
    try:
        if args.fixture_only:
            real_values=(args.execution_results_root,args.r3_lock_dir,args.raw_execution_commit,
                         args.raw_execution_tag,args.verifier_code_commit,args.output_dir)
            if any(value is not None for value in real_values) or args.confirm_read_frozen_real_results:
                raise IndependentPostrunVerificationError("fixture-only mode cannot accept real-result arguments")
            report=fixture_only_self_test()
        else:
            required={
                "repository_root":args.repository_root,"execution_results_root":args.execution_results_root,
                "r3_lock_dir":args.r3_lock_dir,"raw_execution_commit":args.raw_execution_commit,
                "raw_execution_tag":args.raw_execution_tag,"verifier_code_commit":args.verifier_code_commit,
                "output_dir":args.output_dir,
            }
            missing=sorted(name for name,value in required.items() if value is None)
            if missing: raise IndependentPostrunVerificationError(f"missing real-mode arguments: {missing}")
            report=verify_frozen_real_results(
                repository=args.repository_root,execution_results_root=args.execution_results_root,
                r3_lock_dir=args.r3_lock_dir,raw_execution_commit=args.raw_execution_commit,
                raw_execution_tag=args.raw_execution_tag,verifier_code_commit=args.verifier_code_commit,
                output_dir=args.output_dir,
                confirm_read_frozen_real_results=args.confirm_read_frozen_real_results,
            )
    except IndependentPostrunVerificationError as error:
        print(json.dumps({"status":"FAIL","pass":False,"error":str(error)},sort_keys=True))
        return 1
    print(json.dumps(report,sort_keys=True))
    return 0 if report.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
