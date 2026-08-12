#!/usr/bin/env python3
"""Run the twelve fail-closed Boreas v2 Stage-1 verifier tamper cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.io import (  # noqa: E402
    canonical_json_bytes,
    compact_sha256,
    sha256_file,
)


Mutation = Callable[[Path], None]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _mutate_json(root: Path, name: str, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / name
    value = _read(path)
    mutation(value)
    _write(path, value)


def _rehash_outer_closure(root: Path) -> None:
    manifest_path = root / "frozen_manifest.json"
    manifest = _read(manifest_path)
    for row in manifest["payload"]:
        path = root / row["path"]
        row["sha256"] = sha256_file(path)
        row["size_bytes"] = path.stat().st_size
    manifest["eligibility_sha256"] = sha256_file(
        root / "public_data_v2_stage1_eligibility.json"
    )
    manifest.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = compact_sha256(manifest)
    _write(manifest_path, manifest)
    names = sorted(path.name for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _cases() -> list[tuple[str, Mutation]]:
    return [
        (
            "v1_boreas_r02_changed_to_pass",
            lambda root: _mutate_json(
                root,
                "boreas_v2_requalification_from_frozen_evidence.json",
                lambda value: value["v1_conclusion"].__setitem__("R02", "PASS"),
            ),
        ),
        (
            "extrinsic_lidar_assistance_hidden",
            lambda root: _mutate_json(
                root,
                "boreas_v2_requalification_from_frozen_evidence.json",
                lambda value: value["static_extrinsic"].__setitem__(
                    "calibration_used_lidar_point_clouds", False
                ),
            ),
        ),
        (
            "unknown_extrinsic_uncertainty_replaced_with_zero",
            lambda root: _mutate_json(
                root,
                "boreas_v2_requalification_from_frozen_evidence.json",
                lambda value: value["static_extrinsic"].__setitem__("uncertainty", 0),
            ),
        ),
        (
            "minimum_covered_duration_changed",
            lambda root: _mutate_json(
                root,
                "public_data_external_validation_pair_selection_contract_v2.json",
                lambda value: value["eligibility_thresholds"].__setitem__(
                    "MIN_TOTAL_COVERED_DURATION_S", 151.0
                ),
            ),
        ),
        (
            "minimum_coverage_fraction_changed",
            lambda root: _mutate_json(
                root,
                "public_data_external_validation_pair_selection_contract_v2.json",
                lambda value: value["eligibility_thresholds"].__setitem__(
                    "MIN_COVERAGE_FRACTION", 0.61
                ),
            ),
        ),
        (
            "minimum_five_second_intervals_changed",
            lambda root: _mutate_json(
                root,
                "public_data_external_validation_pair_selection_contract_v2.json",
                lambda value: value["eligibility_thresholds"].__setitem__(
                    "MIN_ELIGIBLE_NONOVERLAPPING_5S_INTERVALS", 31
                ),
            ),
        ),
        (
            "primary_pair_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_pair_selection.json",
                lambda value: value.__setitem__("PRIMARY_PAIR", value["RESERVE_PAIR_1"]),
            ),
        ),
        (
            "reserve_pair_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_pair_selection.json",
                lambda value: value.__setitem__("RESERVE_PAIR_1", value["RESERVE_PAIR_2"]),
            ),
        ),
        (
            "gt_evidence_checksum_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_evidence_reuse_manifest.json",
                lambda value: value["files"][0].__setitem__("sha256", "0" * 64),
            ),
        ),
        (
            "lidar_payload_count_falsely_nonzero",
            lambda root: _mutate_json(
                root,
                "NO_LIDAR_PAYLOAD_ATTESTATION.json",
                lambda value: value.__setitem__("downloaded_lidar_payload_count", 1),
            ),
        ),
        (
            "measurement_mainline_falsely_authorized",
            lambda root: _mutate_json(
                root,
                "public_data_v2_stage1_eligibility.json",
                lambda value: value.__setitem__("MEASUREMENT_PAPER_MAINLINE_AUTHORIZED", True),
            ),
        ),
        (
            "missing_external_qualification_falsely_passed",
            lambda root: _mutate_json(
                root,
                "test_baseline_status.json",
                lambda value: value["external_qualification_baseline"].update(
                    {"failed": 0, "passed": 1, "status": "PASS"}
                ),
            ),
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "zero_perturbation_data/boreas_stage1_v1",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=REPOSITORY
        / "frozen_assets/public_data_external_validation_v2_boreas_stage1",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    results: list[dict[str, Any]] = []
    for case_name, mutation in _cases():
        with tempfile.TemporaryDirectory(prefix=f"zprm-v2-tamper-{case_name}-") as temporary:
            candidate = Path(temporary) / "closure"
            shutil.copytree(arguments.frozen_root, candidate)
            mutation(candidate)
            _rehash_outer_closure(candidate)
            command = [
                sys.executable,
                str(arguments.repository_root / "scripts/verify_boreas_external_v2_stage1.py"),
                "--repository-root",
                str(arguments.repository_root),
                "--data-root",
                str(arguments.data_root),
                "--runtime-root",
                str(candidate),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            results.append(
                {
                    "case": case_name,
                    "nonzero_exit": completed.returncode != 0,
                    "returncode": completed.returncode,
                }
            )
    report = {
        "all_tamper_cases_rejected": all(row["nonzero_exit"] for row in results),
        "lidar_bin_test_file_created": False,
        "registration_execution_count": 0,
        "results": results,
        "tamper_case_count": len(results),
    }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["all_tamper_cases_rejected"] and len(results) == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
