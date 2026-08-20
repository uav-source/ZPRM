#!/usr/bin/env python3
"""Run the FMB1 pre-registration ingest in explicitly separated safe phases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY, REPOSITORY / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from phase_a_harness.mid360_pilot.bag_reader import sha256_file  # noqa: E402


RUNTIME_DEFAULT = (
    REPOSITORY / "zero_perturbation_runtime/mid360_formal_batch1_ingest_v1"
)
RESULTS_DEFAULT = REPOSITORY / "results/mid360_formal_batch1"
CONFIG_DEFAULT = REPOSITORY / "configs/mid360_pilot_config.json"

SOURCE_BINDING_PATHS = (
    "experiments/mid360_formal_batch1/preregistration.yaml",
    "experiments/mid360_formal_batch1/protocol.py",
    "experiments/mid360_formal_batch1/preregistration_acquisition.py",
    "experiments/mid360_formal_batch1/preregistration_assets.py",
    "experiments/mid360_formal_batch1/preregistration_finalize.py",
    "experiments/mid360_formal_batch1/preregistration_firewall.py",
    "experiments/mid360_formal_batch1/preregistration_verify.py",
    "experiments/mid360_formal_batch1/preregistration_deep_verify_ros.py",
    "experiments/mid360_formal_batch1/preregistration_deep_verify_geometry.py",
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
    "tools/mid360_formal_batch1/ingest_existing_batch.py",
    "tools/mid360_formal_batch1/verify_pre_registration.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite different stage output: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _source_bindings(repository: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_BINDING_PATHS:
        path = repository / relative
        if path.is_file():
            output[relative] = sha256_file(path)
    return output


def _archive_for_safe_refresh(args: argparse.Namespace) -> None:
    """Archive and authenticate our provisional, still-unauthorized outputs."""

    from experiments.mid360_formal_batch1.preregistration_finalize import REFRESH_ENV

    status = _load_json(RESULTS_DEFAULT / "formal_icp_status.json")
    readiness = _load_json(args.results_dir / "fmb1_pre_registration_readiness.json")
    if status.get("FORMAL_ICP_UNLOCKED") is not False:
        raise RuntimeError("cannot refresh after formal ICP unlock")
    if readiness.get("FORMAL_REGISTRATION_AUTHORIZED") is not False:
        raise RuntimeError("cannot refresh an authorized formal result")
    if readiness.get("FMB1_PRE_REGISTRATION_DATA_READY") is not False:
        raise RuntimeError("refresh is restricted to a failed pre-registration freeze")
    checksum_path = args.results_dir / "SHA256SUMS"
    rows: list[dict[str, Any]] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or len(digest) != 64:
            raise RuntimeError("unsafe or malformed SHA256SUMS entry")
        path = args.results_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"refresh input is not a regular file: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"refresh input checksum mismatch: {path}")
        rows.append({"name": name, "sha256": actual, "bytes": path.stat().st_size})
    archive = (
        args.runtime_dir
        / "output_revisions"
        / "pre_deep_verifier_refinalize_v1"
    )
    if archive.exists():
        raise RuntimeError(f"refresh archive already exists: {archive}")
    archive.mkdir(parents=True)
    copied: set[str] = set()
    for row in rows:
        name = str(row["name"])
        shutil.copy2(args.results_dir / name, archive / name)
        copied.add(name)
    for name in (
        "SHA256SUMS",
        "fmb1_verification_report.json",
        "fmb1_deep_verification_report.json",
    ):
        source = args.results_dir / name
        if source.is_file() and name not in copied:
            shutil.copy2(source, archive / name)
            copied.add(name)
    _write_json_once(
        archive / "ARCHIVE_MANIFEST.json",
        {
            "schema": "mid360_fmb1_provisional_output_archive_v1",
            "reason": "REFINALIZE_AFTER_DEEP_VERIFIER_SOURCE_SET_COMPLETED",
            "source_results_dir": str(args.results_dir.resolve()),
            "files": rows,
            "formal_icp_unlocked": False,
            "formal_registration_authorized": False,
        },
    )
    old_verifier = args.results_dir / "fmb1_verification_report.json"
    if old_verifier.is_file():
        archived_verifier = archive / old_verifier.name
        if not archived_verifier.is_file() or sha256_file(archived_verifier) != sha256_file(old_verifier):
            raise RuntimeError("independent verifier report was not archived exactly")
        old_verifier.unlink()
    os.environ[REFRESH_ENV] = "1"


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_acquisition import (
        run_acquisition,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    config = _load_json(args.config)
    with NoRegistrationGuard() as guard:
        payload = run_acquisition(REPOSITORY, args.bags_dir, config)
    report = guard.report()
    payload["registration_guard"] = report
    payload["source_bindings"] = _source_bindings(REPOSITORY)
    _write_json_once(args.runtime_dir / "acquisition_manifest.json", payload)
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
    acquisition = _load_json(args.runtime_dir / "acquisition_manifest.json")
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
    manifest = args.runtime_dir / "asset_manifest.json"
    _write_json_once(manifest, payload)
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
    assets = _load_json(args.runtime_dir / "asset_manifest.json")
    config = _load_json(args.config)
    with NoRegistrationGuard() as guard:
        payload = analyze_geometry_only(
            assets,
            runtime_dir=args.runtime_dir,
            config=config,
            write_manifest=False,
        )
    payload["registration_guard"] = guard.report()
    manifest = args.runtime_dir / "geometry_only_manifest.json"
    _write_json_once(manifest, payload)
    return payload


def _run_verify_ros(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_deep_verify_ros import (
        verify_ros_evidence,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    acquisition = _load_json(args.runtime_dir / "acquisition_manifest.json")
    assets = _load_json(args.runtime_dir / "asset_manifest.json")
    config = _load_json(args.config)
    with NoRegistrationGuard() as guard:
        payload = verify_ros_evidence(acquisition, assets, config)
    payload["registration_guard"] = guard.report()
    _write_json_once(args.runtime_dir / "deep_ros_verification.json", payload)
    return payload


def _run_verify_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_deep_verify_geometry import (
        verify_geometry_evidence,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        NoRegistrationGuard,
        assert_static_scope_safe,
    )

    assert_static_scope_safe(REPOSITORY)
    assets = _load_json(args.runtime_dir / "asset_manifest.json")
    geometry = _load_json(args.runtime_dir / "geometry_only_manifest.json")
    config = _load_json(args.config)
    with NoRegistrationGuard() as guard:
        payload = verify_geometry_evidence(assets, geometry, config)
    payload["registration_guard"] = guard.report()
    _write_json_once(args.runtime_dir / "deep_geometry_verification.json", payload)
    ros_report_path = args.runtime_dir / "deep_ros_verification.json"
    ros_report = _load_json(ros_report_path) if ros_report_path.is_file() else None
    combined = {
        "schema": "mid360_fmb1_deep_verification_report_v1",
        "status": (
            "PASS"
            if ros_report is not None
            and ros_report.get("status") == "PASS"
            and payload.get("status") == "PASS"
            else "FAIL"
        ),
        "deep_ros": ros_report,
        "deep_geometry": payload,
        "FORMAL_REGISTRATION_AUTHORIZED": False,
        "actual_registration_trials": 0,
    }
    _write_json_once(
        args.results_dir / "fmb1_deep_verification_report.json", combined
    )
    return combined


def _run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.mid360_formal_batch1.preregistration_finalize import (
        finalize_pre_registration,
    )
    from experiments.mid360_formal_batch1.preregistration_firewall import (
        assert_static_scope_safe,
        build_no_icp_attestation,
    )

    static_report = assert_static_scope_safe(REPOSITORY)
    acquisition = _load_json(args.runtime_dir / "acquisition_manifest.json")
    assets = _load_json(args.runtime_dir / "asset_manifest.json")
    geometry = _load_json(args.runtime_dir / "geometry_only_manifest.json")
    deep_report_path = args.results_dir / "fmb1_deep_verification_report.json"
    deep_verification = (
        _load_json(deep_report_path) if deep_report_path.is_file() else None
    )
    stage_reports = [
        acquisition.get("registration_guard", {}),
        assets.get("registration_guard", {}),
        geometry.get("registration_guard", {}),
    ]
    for name in ("deep_ros_verification.json", "deep_geometry_verification.json"):
        path = args.runtime_dir / name
        if path.is_file():
            stage_reports.append(_load_json(path).get("registration_guard", {}))
    attestation = build_no_icp_attestation(
        repository=REPOSITORY,
        stage_reports=stage_reports,
        static_report=static_report,
    )
    return finalize_pre_registration(
        acquisition,
        assets,
        geometry,
        attestation,
        repository=REPOSITORY,
        results_dir=args.results_dir,
        source_bindings=_source_bindings(REPOSITORY),
        deep_verification=deep_verification,
    )


def main() -> int:
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
    parser.add_argument("--bags-dir", type=Path, default=REPOSITORY / "bags")
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--refresh-unregistered-results",
        action="store_true",
        help="archive and atomically refresh only an existing failed, unauthorized preregistration output",
    )
    args = parser.parse_args()
    os.environ["NO_FORMAL_REGISTRATION"] = "true"
    os.environ["ZPRM_REAL_DATA_PREP_NO_REGISTRATION"] = "1"
    if args.refresh_unregistered_results:
        if args.phase != "finalize":
            parser.error("--refresh-unregistered-results is valid only for finalize")
        _archive_for_safe_refresh(args)
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
        "status": payload.get("status", "PASS"),
        "bag_count": len(payload.get("raw_bags", [])),
        "station_count": len(payload.get("stations", [])),
        "target_count": payload.get("target_count"),
        "snapshot_count": payload.get("snapshot_count"),
        "scene_count": len(payload.get("scene_summaries", [])),
        "FMB1_PRE_REGISTRATION_DATA_READY": payload.get(
            "FMB1_PRE_REGISTRATION_DATA_READY"
        ),
        "FORMAL_REGISTRATION_AUTHORIZED": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.phase.startswith("verify-") and payload.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
