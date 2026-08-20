#!/usr/bin/env python3
"""Fixture qualification or separately-authorized future formal analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
for item in (REPOSITORY, REPOSITORY / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from experiments.mid360_formal_batch1.locked_analysis.formal_firewall_v1 import (  # noqa: E402
    FormalReadRequest,
    FormalResultFirewallError,
    load_frozen_formal_attempts,
    reject_ambiguous_mode,
)
from experiments.mid360_formal_batch1.locked_analysis.locked_analysis_v1 import (  # noqa: E402
    analyze_attempts,
    run_fixture_analysis,
    write_analysis_outputs,
)


DEFAULT_PREPARATION = REPOSITORY / (
    "results/mid360_formal_batch1/"
    "zero_perturbation_locked_analysis_preparation_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--formal-results-root", type=Path)
    parser.add_argument("--postrun-verification-root", type=Path)
    parser.add_argument("--analysis-lock-dir", type=Path)
    parser.add_argument("--analysis-code-commit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-read-frozen-formal-results", action="store_true")
    parser.add_argument("--preparation-dir", type=Path, default=DEFAULT_PREPARATION)
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"refusing to overwrite differing evidence: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _fixture(preparation_dir: Path) -> int:
    _, qualification = run_fixture_analysis()
    attestation = {
        "schema": "mid360_fmb1_locked_analysis_no_real_result_attestation_v1",
        "status": "PASS", "pass": True,
        "real_formal_result_files_read": 0,
        "real_translation_values_read": 0,
        "real_rotation_values_read": 0,
        "real_turnover_values_read": 0,
        "real_scene_aggregations": 0,
        "real_weak_rich_comparisons": 0,
        "real_exact_p_values": 0,
        "real_centered_permutation_p_values": 0,
        "registration_backend_calls": 0,
        "REAL_SCIENTIFIC_ANALYSIS_EXECUTED": False,
        "REAL_SCIENTIFIC_ANALYSIS_AUTHORIZED": False,
    }
    _atomic_json(preparation_dir / "locked_analysis_fixture_qualification.json", qualification)
    _atomic_json(preparation_dir / "locked_analysis_no_real_result_attestation.json", attestation)
    for key in (
        "FIXTURE_ANALYSIS_PASS", "REAL_FORMAL_RESULT_FILES_READ",
        "REAL_SCIENTIFIC_AGGREGATION_COUNT", "REAL_WEAK_RICH_COMPARISON_COUNT",
        "REAL_P_VALUE_COUNT",
    ):
        print(f"{key}={str(qualification[key]).lower() if isinstance(qualification[key], bool) else qualification[key]}")
    return 0


def _formal(args: argparse.Namespace) -> int:
    required = {
        "formal_results_root": args.formal_results_root,
        "postrun_verification_root": args.postrun_verification_root,
        "analysis_lock_dir": args.analysis_lock_dir,
        "analysis_code_commit": args.analysis_code_commit,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise FormalResultFirewallError(f"formal mode arguments missing: {missing}")
    request = FormalReadRequest(
        repository=REPOSITORY, formal_results_root=args.formal_results_root,
        postrun_verification_root=args.postrun_verification_root,
        analysis_lock_dir=args.analysis_lock_dir,
        analysis_code_commit=args.analysis_code_commit, output_dir=args.output_dir,
        confirm_read_frozen_formal_results=args.confirm_read_frozen_formal_results,
    )
    result_schema = json.loads((REPOSITORY / (
        "experiments/mid360_formal_batch1/zero_perturbation_trial_result_schema_v1_1.json"
    )).read_text(encoding="utf-8"))
    attempts, boundary = load_frozen_formal_attempts(request, result_schema)
    plan_payload = json.loads((REPOSITORY / (
        "experiments/mid360_formal_batch1/zero_perturbation_trial_plan_v1_1.json"
    )).read_text(encoding="utf-8"))
    plan_rows = plan_payload["rows"]
    output = analyze_attempts(
        plan_rows, attempts, fixture_only=False,
        analysis_provenance={
            "mode": "FORMAL_FROZEN_RESULTS", "artificial_fixture": False,
            "formal_result_files_read": boundary["formal_result_file_count"],
            "analysis_lock_sha256": boundary["analysis_lock_sha256"],
            "analysis_code_commit": boundary["analysis_code_commit"],
            "authorization_sha256": boundary["authorization_sha256"],
        },
    )
    write_analysis_outputs(output, args.output_dir)
    print("FORMAL_ANALYSIS_COMPLETE=true")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    formal_values = (
        args.formal_results_root, args.postrun_verification_root,
        args.analysis_lock_dir, args.analysis_code_commit, args.output_dir,
    )
    mode = reject_ambiguous_mode(
        fixture_only=args.fixture_only,
        formal_arguments_present=any(value is not None for value in formal_values),
        confirmed=args.confirm_read_frozen_formal_results,
    )
    return _fixture(args.preparation_dir) if mode == "FIXTURE_ONLY" else _formal(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormalResultFirewallError, RuntimeError, ValueError, OSError) as exc:
        print(f"LOCKED_ANALYSIS_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
