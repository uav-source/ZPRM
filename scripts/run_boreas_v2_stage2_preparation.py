#!/usr/bin/env python3
"""Run resumable Boreas v2 Stage-2 map preparation (never registration)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from phase_a_harness.real_data_preparation.boreas_v2_stage2_authorization import (  # noqa: E402
    EXPECTED_ALLOWLIST_SHA256,
    EXPECTED_STORAGE_BUDGET_SHA256,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_preprocessing import (  # noqa: E402
    BoreasLidarPoseIndex,
    load_t_applanix_lidar,
)
from phase_a_harness.real_data_preparation.boreas_v2_stage2_runner import (  # noqa: E402
    BoreasV2Stage2Runner,
    BoreasV2Stage2RunnerConfig,
    BoreasV2Stage2RunnerDependencies,
    ProductionMapPreprocessor,
    ProductionRemoteMetadataProvider,
)
from phase_a_harness.real_data_preparation.guard import (  # noqa: E402
    NoRegistrationGuard,
)
from phase_a_harness.real_data_preparation.io import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)

try:  # Bind installed Open3D registration entrypoints for the guard lifetime.
    import open3d as _open3d  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - source-only test environments
    _open3d = None


DEFAULT_AWS = Path(
    "/home/lj/zero_perturbation_data/boreas_stage1_v1/"
    "tools/awscli-venv/bin/aws"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare/freeze the Boreas Stage-2 target map under the active "
            "NoRegistrationGuard. This command never runs ICP or a backend."
        )
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--monitored-disk-path", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--aws-executable", type=Path, default=DEFAULT_AWS)
    parser.add_argument(
        "--phase",
        choices=("map-ingest", "finalize-map", "phase1"),
        default="phase1",
    )
    return parser


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract_status") != "FROZEN":
        raise RuntimeError("preprocessing contract is absent or not frozen")
    return value


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    runtime_root = args.runtime_root.resolve(strict=True)
    temporary_root = args.temporary_root.resolve(strict=True)
    monitored = args.monitored_disk_path.resolve(strict=True)
    contract_path = repository / "protocols/boreas_v2_stage2_preprocessing_contract.json"
    contract = _load_contract(contract_path)
    pair = contract["primary_pair"]
    map_sequence = str(pair["map_sequence_id"])
    map_pose_path = (
        data_root / "stage1_payload" / map_sequence / "applanix" / "lidar_poses.csv"
    )
    extrinsic_path = (
        data_root
        / "stage1_payload"
        / map_sequence
        / "calib"
        / "T_applanix_lidar.txt"
    )
    pose_index = BoreasLidarPoseIndex.from_csv(
        map_pose_path,
        sequence_id=map_sequence,
        expected_sha256=str(pair["map_lidar_pose_sha256"]),
    )
    # Authenticate the static calibration even though the published exact
    # lidar-pose rows already materialize T_ENU_lidar.
    load_t_applanix_lidar(
        extrinsic_path,
        expected_sha256=str(pair["static_t_applanix_lidar_sha256"]),
    )

    config = BoreasV2Stage2RunnerConfig(
        repository=repository,
        data_root=data_root,
        runtime_root=runtime_root,
        temporary_root=temporary_root,
        monitored_disk_path=monitored,
        authorization_path=args.authorization.resolve(strict=True),
        allowlist_path=(
            repository
            / "frozen_assets/public_data_external_validation_v2_boreas_stage1/"
            "boreas_v2_stage2_download_allowlist.csv"
        ),
        expected_allowlist_sha256=EXPECTED_ALLOWLIST_SHA256,
        disk_budget_path=(
            repository
            / "frozen_assets/boreas_v2_stage2_storage_optimization/"
            "boreas_v2_stage2_disk_budget_optimized.json"
        ),
        expected_disk_budget_sha256=EXPECTED_STORAGE_BUDGET_SHA256,
        preprocessing_contract_path=contract_path,
        aws_executable=args.aws_executable.resolve(strict=True),
        reducer_resource_plan_path=(
            runtime_root / "checkpoints/reducer_resource_plan.json"
        ),
    )
    os.environ.setdefault("ZPRM_REAL_DATA_PREP_NO_REGISTRATION", "1")
    with NoRegistrationGuard(open3d_module=_open3d) as guard:
        dependencies = BoreasV2Stage2RunnerDependencies(
            metadata_provider=ProductionRemoteMetadataProvider(
                aws_executable=config.aws_executable,
                bucket=config.bucket,
            ),
            map_preprocessor=ProductionMapPreprocessor(
                pose_index=pose_index,
                preprocessing_contract_sha256=sha256_file(contract_path),
                gt_sha256=str(pair["map_lidar_pose_sha256"]),
                extrinsic_sha256=str(pair["static_t_applanix_lidar_sha256"]),
            ),
            no_registration_guard=guard,
        )
        with BoreasV2Stage2Runner(config, dependencies) as runner:
            result: dict[str, Any] = {
                "phase": args.phase,
                "runner_status": "PHASE1_IN_PROGRESS_NOT_STAGE2_READY",
            }
            if args.phase in {"map-ingest", "phase1"}:
                result["map_ingest"] = runner.run_map_ingest().__dict__
            if args.phase in {"finalize-map", "phase1"}:
                assert runner.reducer_resource_plan is not None
                result["target_map"] = runner.finalize_target_map(
                    runner.reducer_resource_plan
                )
                result["runner_status"] = (
                    "PHASE1_MAP_FROZEN_QUERY_NOT_STARTED_NOT_STAGE2_READY"
                )
            print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
