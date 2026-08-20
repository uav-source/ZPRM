#!/usr/bin/env python3
"""Run the W02 attempt-2 preparation in explicit no-registration phases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from phase_a_harness.mid360_pilot.bag_reader import sha256_file  # noqa: E402

from experiments.mid360_formal_batch1.w02_attempt2_outputs import (  # noqa: E402
    write_acquisition_evidence,
    write_asset_evidence,
    write_geometry_evidence,
    write_json_once,
    write_sha256sums,
)


RUNTIME_DEFAULT = (
    REPOSITORY
    / "zero_perturbation_runtime/mid360_formal_batch1_w02_attempt2_v1"
)
RESULTS_DEFAULT = REPOSITORY / "results/mid360_formal_batch1/w02_attempt2_v1"
CONFIG_DEFAULT = REPOSITORY / "configs/mid360_pilot_config.json"
SOURCE_BINDING_PATHS = (
    "experiments/mid360_formal_batch1/preregistration.yaml",
    "experiments/mid360_formal_batch1/analysis_protocol.md",
    "experiments/mid360_formal_batch1/protocol.py",
    "experiments/mid360_formal_batch1/preregistration_acquisition.py",
    "experiments/mid360_formal_batch1/preregistration_assets.py",
    "experiments/mid360_formal_batch1/preregistration_firewall.py",
    "experiments/mid360_formal_batch1/preregistration_deep_verify_ros.py",
    "experiments/mid360_formal_batch1/preregistration_deep_verify_geometry.py",
    "experiments/mid360_formal_batch1/w04_replacement.py",
    "experiments/mid360_formal_batch1/w02_attempt2.py",
    "experiments/mid360_formal_batch1/w02_attempt2_deep_verify.py",
    "experiments/mid360_formal_batch1/w02_attempt2_outputs.py",
    "tools/mid360_formal_batch1/process_w02_attempt2.py",
    "configs/mid360_pilot_config.json",
    "src/phase_a_harness/mid360_pilot/bag_reader.py",
    "src/phase_a_harness/mid360_pilot/lidar_adapter.py",
    "src/phase_a_harness/mid360_pilot/imu_audit.py",
    "src/phase_a_harness/mid360_pilot/static_map.py",
    "src/phase_a_harness/mid360_pilot/split.py",
    "src/phase_a_harness/mid360_pilot/pilot_geometry.py",
    "src/phase_a_harness/common_association_analysis.py",
    "src/phase_a_harness/real_data_preparation/boreas_v2_stage2_selection.py",
    "frozen_assets/backend_parameter_contract.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _source_bindings(repository: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_BINDING_PATHS:
        path = repository / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source binding is missing or unsafe: {path}")
        output[relative] = sha256_file(path)
    return output


def _guarded() -> tuple[Any, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    return NoRegistrationGuard, assert_static_scope_safe


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.w02_attempt2 import (
        run_w02_attempt2_acquisition,
    )

    guard_type, _ = _guarded()
    with guard_type() as guard:
        payload = run_w02_attempt2_acquisition(REPOSITORY, args.config)
    payload["registration_guard"] = guard.report()
    payload["source_bindings"] = _source_bindings(REPOSITORY)
    write_json_once(args.runtime_dir / "w02_attempt2_acquisition_manifest.json", payload)
    write_acquisition_evidence(payload, args.results_dir)
    return payload


def _run_assets(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_assets import (
        materialize_registration_free_assets,
    )

    acquisition = _load_json(
        args.runtime_dir / "w02_attempt2_acquisition_manifest.json"
    )
    if acquisition.get("W02_ATTEMPT2_ACQUISITION_PASS") is not True:
        raise RuntimeError("W02 attempt-2 acquisition did not pass")
    guard_type, _ = _guarded()
    with guard_type() as guard:
        payload = materialize_registration_free_assets(
            acquisition,
            runtime_dir=args.runtime_dir,
            config=_load_json(args.config),
            source_bindings=_source_bindings(REPOSITORY),
            write_manifest=False,
        )
    payload["registration_guard"] = guard.report()
    payload["attempt"] = 2
    write_json_once(args.runtime_dir / "w02_attempt2_asset_manifest.json", payload)
    write_asset_evidence(payload, args.results_dir)
    return payload


def _run_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_assets import (
        analyze_geometry_only,
    )

    assets = _load_json(args.runtime_dir / "w02_attempt2_asset_manifest.json")
    if assets.get("target_count") != 3 or assets.get("snapshot_count") != 30:
        raise RuntimeError("W02 attempt-2 assets are not exact 3/30")
    guard_type, _ = _guarded()
    with guard_type() as guard:
        payload = analyze_geometry_only(
            assets,
            runtime_dir=args.runtime_dir,
            config=_load_json(args.config),
            write_manifest=False,
        )
    payload["registration_guard"] = guard.report()
    payload["attempt"] = 2
    write_json_once(
        args.runtime_dir / "w02_attempt2_geometry_only_manifest.json", payload
    )
    write_geometry_evidence(payload, args.results_dir)
    return payload


def _run_verify_ros(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.w02_attempt2_deep_verify import (
        verify_w02_attempt2_ros_evidence,
    )

    acquisition = _load_json(
        args.runtime_dir / "w02_attempt2_acquisition_manifest.json"
    )
    assets = _load_json(args.runtime_dir / "w02_attempt2_asset_manifest.json")
    guard_type, _ = _guarded()
    with guard_type() as guard:
        payload = verify_w02_attempt2_ros_evidence(acquisition, assets, args.config)
    payload["registration_guard"] = guard.report()
    write_json_once(
        args.runtime_dir / "w02_attempt2_deep_ros_verification.json", payload
    )
    write_json_once(
        args.results_dir / "w02_attempt2_deep_ros_verification.json", payload
    )
    return payload


def _run_verify_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.w02_attempt2_deep_verify import (
        verify_w02_attempt2_geometry_evidence,
    )

    assets = _load_json(args.runtime_dir / "w02_attempt2_asset_manifest.json")
    geometry = _load_json(
        args.runtime_dir / "w02_attempt2_geometry_only_manifest.json"
    )
    guard_type, _ = _guarded()
    with guard_type() as guard:
        payload = verify_w02_attempt2_geometry_evidence(
            assets, geometry, args.config
        )
    payload["registration_guard"] = guard.report()
    write_json_once(
        args.runtime_dir / "w02_attempt2_deep_geometry_verification.json", payload
    )
    write_json_once(
        args.results_dir / "w02_attempt2_deep_geometry_verification.json", payload
    )
    return payload


def _run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        assert_static_scope_safe,
        build_no_icp_attestation,
    )

    acquisition = _load_json(
        args.runtime_dir / "w02_attempt2_acquisition_manifest.json"
    )
    assets = _load_json(args.runtime_dir / "w02_attempt2_asset_manifest.json")
    geometry = _load_json(
        args.runtime_dir / "w02_attempt2_geometry_only_manifest.json"
    )
    ros_verify = _load_json(
        args.runtime_dir / "w02_attempt2_deep_ros_verification.json"
    )
    geometry_verify = _load_json(
        args.runtime_dir / "w02_attempt2_deep_geometry_verification.json"
    )
    reports = [
        row.get("registration_guard", {})
        for row in (acquisition, assets, geometry, ros_verify, geometry_verify)
    ]
    attestation = build_no_icp_attestation(
        REPOSITORY,
        stage_reports=reports,
        static_report=assert_static_scope_safe(REPOSITORY),
    )
    attestation["FORMAL_LOCK_ISSUED"] = False
    attestation["FORMAL_ICP_UNLOCKED"] = False
    write_json_once(args.results_dir / "NO_ICP_ATTESTATION.json", attestation)
    scene = geometry["scene_summaries"][0]
    admitted = bool(
        scene.get("final_geometry_class") == "WEAK"
        and scene.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
    )
    readiness = {
        "schema": "mid360_fmb1_w02_attempt2_readiness_v1",
        "status": "PASS" if admitted else "FAIL",
        "scene_id": "FMB1_W02",
        "attempt": 2,
        "W02_ATTEMPT2_ACQUISITION_PASS": acquisition.get(
            "W02_ATTEMPT2_ACQUISITION_PASS"
        )
        is True,
        "W02_ATTEMPT2_FINAL_GEOMETRY_CLASS": scene.get(
            "final_geometry_class"
        ),
        "W02_ATTEMPT2_ADMISSION_PASS": admitted,
        "geometry_scene_medians": {
            "normalized_lambda_min_trans": scene.get(
                "median_normalized_lambda_min_trans"
            ),
            "condition_number_trans": scene.get("median_condition_number_trans"),
            "spectral_entropy_trans": scene.get("median_spectral_entropy_trans"),
        },
        "target_count": assets.get("target_count"),
        "snapshot_count": assets.get("snapshot_count"),
        "query_contribution_to_every_target": assets.get(
            "query_contribution_to_every_target"
        ),
        "deep_ros_verifier_pass": ros_verify.get("status") == "PASS",
        "deep_geometry_verifier_pass": geometry_verify.get("status") == "PASS",
        "NO_ICP_ATTESTATION_PASS": attestation.get("status") == "PASS",
        "correction_reason": "OPERATOR_CONFIRMED_WRONG_SCENE_LOCATION",
        "correction_before_any_icp": True,
        "formal_trial_count_at_correction": 0,
        "failed_attempts_retained": True,
        "failed_raw_data_retained": True,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "source_bindings": _source_bindings(REPOSITORY),
    }
    write_json_once(
        args.results_dir / "w02_attempt2_readiness.json", readiness
    )
    write_sha256sums(args.results_dir)
    return readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "audit",
            "assets",
            "geometry",
            "verify-ros",
            "verify-geometry",
            "finalize",
        ),
    )
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    args = parser.parse_args(argv)
    os.environ["NO_FORMAL_REGISTRATION"] = "true"
    os.environ["ZPRM_REAL_DATA_PREP_NO_REGISTRATION"] = "1"
    os.environ["ZPRM_FMB1_NO_FORMAL_REGISTRATION"] = "1"
    args.runtime_dir = args.runtime_dir.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve(strict=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    action = {
        "audit": _run_audit,
        "assets": _run_assets,
        "geometry": _run_geometry,
        "verify-ros": _run_verify_ros,
        "verify-geometry": _run_verify_geometry,
        "finalize": _run_finalize,
    }[args.phase]
    payload = action(args)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": payload.get("status"),
                "scene_id": "FMB1_W02",
                "attempt": 2,
                "FORMAL_REGISTRATION_AUTHORIZED": False,
                "actual_formal_trials": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.phase.startswith("verify-") and payload.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
