#!/usr/bin/env python3
"""Run eleven re-signed Boreas v2 Stage-2 storage-plan tamper cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parents[1]

Mutation = Callable[[Path], None]


def _canonical_json_bytes(value: Any) -> bytes:
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


def _compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _mutate_json(
    root: Path, name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / name
    value = _read(path)
    mutation(value)
    _write(path, value)


def _rehash_outer_closure(root: Path) -> None:
    """Re-sign the outer closure without repairing the semantic mutation."""

    manifest_path = root / "frozen_manifest.json"
    manifest = _read(manifest_path)
    for row in manifest["payload"]:
        path = root / row["path"]
        row["sha256"] = _sha256_file(path)
        row["size_bytes"] = path.stat().st_size
    manifest.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = _compact_sha256(manifest)
    _write(manifest_path, manifest)
    names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def tamper_cases() -> list[tuple[str, Mutation]]:
    """Return the eleven normative semantic tamper cases from the audit."""

    return [
        (
            "primary_pair_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_stage2_storage_readiness.json",
                lambda value: value["stage1_binding"]["primary_pair"].__setitem__(
                    "query_sequence_id", "boreas-2099-tampered"
                ),
            ),
        ),
        (
            "object_count_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_stage2_storage_readiness.json",
                lambda value: value["stage1_binding"].__setitem__(
                    "allowlist_object_count", 20_060
                ),
            ),
        ),
        (
            "remote_bytes_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_stage2_storage_readiness.json",
                lambda value: value["stage1_binding"].__setitem__(
                    "allowlist_remote_bytes", 104_158_637_471
                ),
            ),
        ),
        (
            "target_copy_count_greater_than_one",
            lambda root: _mutate_json(
                root,
                "stage2_streaming_map_contract.json",
                lambda value: value["content_addressing"].__setitem__(
                    "planned_physical_target_map_copy_count", 2
                ),
            ),
        ),
        (
            "open3d_pcl_source_sha_diverged",
            lambda root: _mutate_json(
                root,
                "canonical_bundle_storage_contract_v2.json",
                lambda value: value.__setitem__("future_pcl_source_sha256", "0" * 64),
            ),
        ),
        (
            "snapshot_count_changed",
            lambda root: _mutate_json(
                root,
                "stage2_streaming_query_contract.json",
                lambda value: value["selection_contract"].__setitem__(
                    "planned_snapshot_count", 99
                ),
            ),
        ),
        (
            "voxel_size_configured_by_storage_planner",
            lambda root: _mutate_json(
                root,
                "stage2_streaming_map_contract.json",
                lambda value: value["scientific_preprocessing_parameters"].__setitem__(
                    "voxel_size_m", 0.20
                ),
            ),
        ),
        (
            "no_lidar_attestation_changed",
            lambda root: _mutate_json(
                root,
                "NO_LIDAR_PAYLOAD_ATTESTATION.json",
                lambda value: value.__setitem__("downloaded_lidar_payload_count", 1),
            ),
        ),
        (
            "disk_budget_arithmetic_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_stage2_disk_budget_optimized.json",
                lambda value: value["modes"][1].__setitem__(
                    "recommended_free_disk_bytes",
                    value["modes"][1]["recommended_free_disk_bytes"] + 1,
                ),
            ),
        ),
        (
            "low_disk_watermark_changed",
            lambda root: _mutate_json(
                root,
                "boreas_v2_stage2_disk_budget_optimized.json",
                lambda value: value["modes"][1].__setitem__(
                    "runtime_low_disk_watermark_bytes",
                    value["modes"][1]["runtime_low_disk_watermark_bytes"] + 1,
                ),
            ),
        ),
        (
            "stage1_manifest_sha_binding_changed",
            lambda root: _mutate_json(
                root,
                "frozen_manifest.json",
                lambda value: value.__setitem__(
                    "stage1_frozen_manifest_file_sha256", "0" * 64
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
        "--runtime-root",
        dest="frozen_root",
        type=Path,
        default=REPOSITORY
        / "frozen_assets/boreas_v2_stage2_storage_optimization",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    results: list[dict[str, Any]] = []
    for case_name, mutation in tamper_cases():
        with tempfile.TemporaryDirectory(
            prefix=f"zprm-boreas-stage2-storage-tamper-{case_name}-"
        ) as temporary:
            candidate = Path(temporary) / "closure"
            shutil.copytree(arguments.frozen_root, candidate)
            mutation(candidate)
            _rehash_outer_closure(candidate)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        arguments.repository_root
                        / "scripts/verify_boreas_v2_stage2_storage_plan.py"
                    ),
                    "--repository-root",
                    str(arguments.repository_root),
                    "--data-root",
                    str(arguments.data_root),
                    "--runtime-root",
                    str(candidate),
                ],
                text=True,
                capture_output=True,
            )
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
    return 0 if report["all_tamper_cases_rejected"] and len(results) == 11 else 1


if __name__ == "__main__":
    raise SystemExit(main())
