#!/usr/bin/env python3
"""Run W04 replacement preparation in explicit no-registration phases."""

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

from experiments.mid360_formal_batch1.w04_outputs import (  # noqa: E402
    write_acquisition_evidence,
    write_asset_evidence,
    write_geometry_evidence,
    write_json_once,
    write_sha256sums,
)


RUNTIME_DEFAULT = (
    REPOSITORY
    / "zero_perturbation_runtime/mid360_formal_batch1_w04_replacement_v1"
)
RESULTS_DEFAULT = REPOSITORY / "results/mid360_formal_batch1/w04_replacement_v1"
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
    "experiments/mid360_formal_batch1/w04_outputs.py",
    "experiments/mid360_formal_batch1/w04_deep_verify.py",
    "tools/mid360_formal_batch1/process_w04_replacement.py",
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


def _write_runtime(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_once(path, payload)


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )
    from experiments.mid360_formal_batch1.w04_replacement import (
        run_w04_acquisition,
    )

    assert_static_scope_safe(REPOSITORY)
    with NoRegistrationGuard() as guard:
        payload = run_w04_acquisition(REPOSITORY, args.config)
    payload["registration_guard"] = guard.report()
    payload["source_bindings"] = _source_bindings(REPOSITORY)
    _write_runtime(args.runtime_dir / "w04_acquisition_manifest.json", payload)
    write_acquisition_evidence(payload, args.results_dir)
    return payload


def _run_assets(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_assets import (
        materialize_registration_free_assets,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    acquisition = _load_json(args.runtime_dir / "w04_acquisition_manifest.json")
    if acquisition.get("W04_ACQUISITION_PASS") is not True:
        raise RuntimeError("W04 acquisition did not pass; asset stage is forbidden")
    config = _load_json(args.config)
    with NoRegistrationGuard() as guard:
        payload = materialize_registration_free_assets(
            acquisition,
            runtime_dir=args.runtime_dir,
            config=config,
            source_bindings=_source_bindings(REPOSITORY),
            write_manifest=False,
        )
    payload["registration_guard"] = guard.report()
    _write_runtime(args.runtime_dir / "w04_asset_manifest.json", payload)
    write_asset_evidence(payload, args.results_dir)
    return payload


def _run_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_assets import (
        analyze_geometry_only,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    assets = _load_json(args.runtime_dir / "w04_asset_manifest.json")
    if int(assets.get("target_count", -1)) != 3 or int(
        assets.get("snapshot_count", -1)
    ) != 30:
        raise RuntimeError("W04 assets are not exact 3 targets / 30 snapshots")
    with NoRegistrationGuard() as guard:
        payload = analyze_geometry_only(
            assets,
            runtime_dir=args.runtime_dir,
            config=_load_json(args.config),
            write_manifest=False,
        )
    payload["registration_guard"] = guard.report()
    _write_runtime(args.runtime_dir / "w04_geometry_only_manifest.json", payload)
    write_geometry_evidence(payload, args.results_dir)
    return payload


def _run_verify_ros(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )
    from experiments.mid360_formal_batch1.w04_deep_verify import (
        verify_w04_ros_evidence,
    )

    assert_static_scope_safe(REPOSITORY)
    acquisition = _load_json(args.runtime_dir / "w04_acquisition_manifest.json")
    assets = _load_json(args.runtime_dir / "w04_asset_manifest.json")
    with NoRegistrationGuard() as guard:
        payload = verify_w04_ros_evidence(acquisition, assets, args.config)
    payload["registration_guard"] = guard.report()
    _write_runtime(args.runtime_dir / "w04_deep_ros_verification.json", payload)
    write_json_once(args.results_dir / "w04_deep_ros_verification.json", payload)
    return payload


def _run_verify_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )
    from experiments.mid360_formal_batch1.w04_deep_verify import (
        verify_w04_geometry_evidence,
    )

    assert_static_scope_safe(REPOSITORY)
    assets = _load_json(args.runtime_dir / "w04_asset_manifest.json")
    geometry = _load_json(args.runtime_dir / "w04_geometry_only_manifest.json")
    with NoRegistrationGuard() as guard:
        payload = verify_w04_geometry_evidence(assets, geometry, args.config)
    payload["registration_guard"] = guard.report()
    _write_runtime(args.runtime_dir / "w04_deep_geometry_verification.json", payload)
    write_json_once(args.results_dir / "w04_deep_geometry_verification.json", payload)
    return payload


def _summary_markdown(readiness: Mapping[str, Any]) -> str:
    medians = readiness.get("geometry_scene_medians", {})
    return "\n".join(
        [
            "# FMB1 W04 replacement readiness",
            "",
            f"- Acquisition PASS: `{str(readiness['FMB1_W04_ACQUISITION_PASS']).lower()}`",
            f"- Geometry class: `{readiness['FMB1_W04_FINAL_GEOMETRY_CLASS']}`",
            f"- Weak admission PASS: `{str(readiness['FMB1_W04_ADMISSION_PASS']).lower()}`",
            f"- normalized lambda-min median: `{medians.get('normalized_lambda_min_trans')}`",
            f"- condition-number median: `{medians.get('condition_number_trans')}`",
            f"- spectral-entropy median: `{medians.get('spectral_entropy_trans')}`",
            "",
            "Formal lock, ICP unlock, registration authorization, and actual trials remain zero/false.",
            "",
        ]
    )


def _run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        assert_static_scope_safe,
        build_no_icp_attestation,
    )

    acquisition = _load_json(args.runtime_dir / "w04_acquisition_manifest.json")
    assets = _load_json(args.runtime_dir / "w04_asset_manifest.json")
    geometry = _load_json(args.runtime_dir / "w04_geometry_only_manifest.json")
    ros_verify = _load_json(args.runtime_dir / "w04_deep_ros_verification.json")
    geometry_verify = _load_json(
        args.runtime_dir / "w04_deep_geometry_verification.json"
    )
    stage_reports = [
        acquisition.get("registration_guard", {}),
        assets.get("registration_guard", {}),
        geometry.get("registration_guard", {}),
        ros_verify.get("registration_guard", {}),
        geometry_verify.get("registration_guard", {}),
    ]
    attestation = build_no_icp_attestation(
        REPOSITORY,
        stage_reports=stage_reports,
        static_report=assert_static_scope_safe(REPOSITORY),
    )
    write_json_once(args.results_dir / "NO_ICP_ATTESTATION.json", attestation)
    scenes = geometry.get("scene_summaries", [])
    if not isinstance(scenes, list) or len(scenes) != 1:
        raise RuntimeError("W04 geometry scene summary is absent")
    scene = scenes[0]
    admission_pass = bool(
        scene.get("final_geometry_class") == "WEAK"
        and scene.get("geometry_admission_status") == "GEOMETRY_ADMITTED"
    )
    medians = {
        "normalized_lambda_min_trans": scene.get(
            "median_normalized_lambda_min_trans"
        ),
        "condition_number_trans": scene.get("median_condition_number_trans"),
        "spectral_entropy_trans": scene.get("median_spectral_entropy_trans"),
    }
    readiness = {
        "schema": "mid360_fmb1_w04_replacement_readiness_v1",
        "status": "PASS" if admission_pass else "REACQUISITION_REQUIRED",
        "FMB1_W04_ACQUISITION_PASS": acquisition.get("W04_ACQUISITION_PASS")
        is True,
        "FMB1_W04_FINAL_GEOMETRY_CLASS": scene.get("final_geometry_class"),
        "FMB1_W04_ADMISSION_PASS": admission_pass,
        "geometry_scene_medians": medians,
        "weak_gate": {
            "normalized_lambda_min_trans_max": 0.12,
            "condition_number_trans_min": 6.0,
            "spectral_entropy_trans_max": 0.80,
            "logic": "ALL",
            "normalized_lambda_min_trans_pass": float(
                medians["normalized_lambda_min_trans"]
            )
            <= 0.12,
            "condition_number_trans_pass": float(medians["condition_number_trans"])
            >= 6.0,
            "spectral_entropy_trans_pass": float(
                medians["spectral_entropy_trans"]
            )
            <= 0.80,
        },
        "target_count": int(assets.get("target_count", -1)),
        "snapshot_count": int(assets.get("snapshot_count", -1)),
        "query_contribution_to_every_target": assets.get(
            "query_contribution_to_every_target"
        ),
        "deep_ros_verifier_pass": ros_verify.get("status") == "PASS",
        "deep_geometry_verifier_pass": geometry_verify.get("status") == "PASS",
        "NO_ICP_ATTESTATION_PASS": attestation.get("status") == "PASS",
        "W02_RETAINED": True,
        "W02_INCLUDED_IN_FINAL_SET": False,
        "W02_REJECTION_REASON": "GEOMETRY_ONLY_INELIGIBLE",
        "replacement_scene_id": "FMB1_W04",
        "RAW_CANDIDATE_BAG_COUNT": 42,
        "FINAL_ADMITTED_BAG_COUNT": 36 if admission_pass else None,
        "REJECTED_BAG_COUNT": 6,
        "FMB1_FINAL_DATASET_READY": False,
        "READY_FOR_ZERO_PERTURBATION_AMENDMENT_ACTIVATION": False,
        "PROPOSED_AMENDMENT_ACTIVE": False,
        "FORMAL_LOCK_ISSUED": False,
        "FORMAL_ICP_UNLOCKED": False,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_open3d_trials": 0,
        "actual_pcl_trials": 0,
        "actual_formal_trials": 0,
        "source_bindings": _source_bindings(REPOSITORY),
    }
    write_json_once(args.results_dir / "w04_replacement_readiness.json", readiness)
    from experiments.mid360_formal_batch1.w04_outputs import _write_once

    _write_once(
        args.results_dir / "w04_replacement_summary.md",
        _summary_markdown(readiness).encode("utf-8"),
    )
    if not admission_pass:
        write_json_once(
            args.results_dir / "REACQUISITION_REQUIRED.json",
            {
                "schema": "mid360_fmb1_w04_reacquisition_required_v1",
                "REACQUISITION_REQUIRED": True,
                "scene_id": "FMB1_W04",
                "reason": scene.get("failure_reason")
                or "W04_NOT_ADMITTED_AS_WEAK",
                "raw_data_retained": True,
                "actual_formal_trials": 0,
            },
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
    payload = {
        "audit": _run_audit,
        "assets": _run_assets,
        "geometry": _run_geometry,
        "verify-ros": _run_verify_ros,
        "verify-geometry": _run_verify_geometry,
        "finalize": _run_finalize,
    }[args.phase](args)
    summary = {
        "phase": args.phase,
        "status": payload.get("status"),
        "W04_ACQUISITION_PASS": payload.get("W04_ACQUISITION_PASS"),
        "target_count": payload.get("target_count"),
        "snapshot_count": payload.get("snapshot_count"),
        "FMB1_W04_FINAL_GEOMETRY_CLASS": payload.get(
            "FMB1_W04_FINAL_GEOMETRY_CLASS"
        ),
        "FMB1_W04_ADMISSION_PASS": payload.get("FMB1_W04_ADMISSION_PASS"),
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_formal_trials": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.phase.startswith("verify-") and payload.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
