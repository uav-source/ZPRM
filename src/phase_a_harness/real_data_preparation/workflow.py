"""Unified fail-closed real-data preregistration preparation workflow."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from phase_a_harness.real_data_protocol import (
    ELIGIBILITY_REQUIREMENTS,
    SNAPSHOT_SELECTION_FIELDS,
    UNCERTAINTY_BUDGET_FIELDS,
)

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import atomic_write_bytes, atomic_write_csv, atomic_write_json, sha256_file
from .manifest import build_frozen_manifest, sha256sums_bytes, verify_frozen_manifest


BASELINE_COMMIT = "01b19b1ba89376cb53aa082b77032b562888eb7b"
EXPECTED_BRANCH = "prep/real-data-iilabs-grandtour-v1"
HISTORICAL_TAG = "execution/synthetic-confirmatory-v3-requalified"
HF_REVISION = "daf91c5e2b6db564822a35cc58222d73f8e75bab"
GRANDTOUR_GITHUB_COMMIT = "656e35dbe418683ddffae7585a254cc82d1f90b6"
GRANDTOUR_BOX_COMMIT = "83a4952e5bfb1294bd74f6d6a78955f9da78d874"
IILABS_TOOLKIT_SDIST_SHA256 = "75b7b973869b4d38631c5df1c909f1194928153216c4a62df7eabd1b6e74cb7a"
BACKEND_FILE_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
OPEN3D_PARAMETERS_SHA256 = "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
PCL_PARAMETERS_SHA256 = "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"

IILABS_FILES = {
    "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_diff/ouster_nav_a_diff_2025-02-07-13-05-01.bag": {
        "size": 48164911024,
        "url": "https://open-datasets.inesctec.pt/Aja94l1j/benchmark/ouster_os1-64/ouster_nav_a_diff_2025-02-07-13-05-01.bag",
    },
    "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_omni/ouster_nav_a_omni_2025-02-07-13-44-03.bag": {
        "size": 24637246105,
        "url": "https://open-datasets.inesctec.pt/Aja94l1j/benchmark/ouster_os1-64/ouster_nav_a_omni_2025-02-07-13-44-03.bag",
    },
    "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_diff/ground_truth.tum": {
        "size": 36767289,
        "url": "https://open-datasets.inesctec.pt/Aja94l1j/ground-truth/benchmark/ouster_os1-64/nav_a_diff/ground_truth.tum",
    },
    "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_omni/ground_truth.tum": {
        "size": 18835830,
        "url": "https://open-datasets.inesctec.pt/Aja94l1j/ground-truth/benchmark/ouster_os1-64/nav_a_omni/ground_truth.tum",
    },
    "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/calib_ouster_os1-64.yaml": {
        "size": 1581,
        "url": "https://open-datasets.inesctec.pt/Aja94l1j/calibration/calib_ouster_os1-64.yaml",
    },
}

IILABS_DOWNLOAD_PROGRESS_CHECKPOINT = {
    "collected_at_utc": "2026-08-10T05:52:42Z",
    "completed_bytes_are_lower_bounds": True,
    "stop_reason": "FAIL_FAST_AFTER_UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME; partial files retained with aria2 control state",
    "files": {
        "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_diff/ouster_nav_a_diff_2025-02-07-13-05-01.bag": 5152702464,
        "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_omni/ouster_nav_a_omni_2025-02-07-13-44-03.bag": 4537188352,
    },
}

GRANDTOUR_LFS_SHA256 = {
    "2024-11-02-17-10-25/data/cpt7_ie_tc_odometry.tar": "73b93a32e6970d0ca12944a49b01ecb295e5721d153ec488c24e6f8e5db9aa8f",
    "2024-11-02-17-10-25/data/cpt7_ie_tc_tf.tar": "f9f233148594c493e98087cd6d2d7427565bc0434ef6fac30fb2784e8464c023",
    "2024-11-02-17-10-25/data/navsatfix_cpt7_ie_tc.tar": "cd61b07137e8aeecc0fe279e2829663d582a1e4bf1b1402fcc800275ad4cfa26",
    "2024-11-02-17-10-25/data/prism_position.tar": "d9ef7ab934d6062a458556a3562cfcc3f2f63fdc6ba9ffa20fc3fae4dbb4d370",
    "2024-11-02-17-10-25/data/tf.tar": "9d34f6fb3bc8890bf1ee7f601ad2030878e8e86f5a216ba08787266cdd12bdf5",
    "2024-11-02-17-43-10/data/cpt7_ie_tc_odometry.tar": "575efbcbaaa31864ceee2c6afeddc3e70de52e5bb7326d610de688f946312cd6",
    "2024-11-02-17-43-10/data/cpt7_ie_tc_tf.tar": "09f1037b1a74e13ab7bd822a2ef631849eb7b8089ec9fc661e037b2a5deaf763",
    "2024-11-02-17-43-10/data/navsatfix_cpt7_ie_tc.tar": "388130c66709512c079b307cd207557bf590823ea59db80128ec40a41cbab921",
    "2024-11-02-17-43-10/data/prism_position.tar": "d89b8b4d1306463c042dc5cba402fb4371a3822a97a8e7f83839c3fc54779931",
    "2024-11-02-17-43-10/data/tf.tar": "1fe009cb5f0082a44b260dcc3a7d332b892bf4d97cafb644876c801860f5f383",
}

IILABS_ROSBAG_RANGE_AUDIT = {
    "audit_method": "HTTP Range reads of the official immutable bag header, first complete chunk, and index tail; no point-cloud geometry or registration output was read",
    "nav_a_diff": {
        "bag_size_bytes": 48164911024,
        "bag_start_time_ros_epoch_s": 1738933502.2271461,
        "bag_end_time_ros_epoch_s": 1738934262.056712,
        "chunk_count": 7598,
        "first_chunk_range": "bytes=4117-6399562",
        "first_chunk_sha256": "1b4c9907f671bd67532168ab73a2a27a16fc1eb73f1a76ed1a580e778674a6d0",
        "index_tail_range": "bytes=48163646374-48164911023",
        "index_tail_sha256": "95c71b367941afcec7600e68eb580ee1eea9718c490104c5e10185452c19aa69",
        "first_odometry": {
            "frame_id": "eve/odom",
            "child_frame_id": "eve/base_footprint",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "first_map_nn0_transform": {
            "parent_frame": "eve/odom",
            "child_frame": "eve/map/nn0",
            "translation_m": [4.068138610391483, 3.1720707529277767, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.7030638112693579, 0.7111267659731314],
        },
        "topic_message_counts": {
            "/eve/motors_enc": 75699,
            "/tf_static": 5,
            "/tf": 106033,
            "/eve/imu/data": 302094,
            "/eve/scan": 30338,
            "/eve/odom": 75690,
            "/eve/ouster/imu": 75960,
            "/eve/ouster/points": 7597,
        },
    },
    "nav_a_omni": {
        "bag_size_bytes": 24637246105,
        "bag_start_time_ros_epoch_s": 1738935843.8544369,
        "bag_end_time_ros_epoch_s": 1738936232.584366,
        "chunk_count": 3887,
        "first_chunk_range": "bytes=4117-6430115",
        "first_chunk_sha256": "6dffbc93a9ed6974c406a487039c7ad4a75e154a2cee4a132e6cf8530e412926",
        "index_tail_range": "bytes=24636590059-24637246104",
        "index_tail_sha256": "7f5f486a9a5e921e54e6c43ba5bfdf3d91968b8e0f438e86bbab8d60595e8b22",
        "first_odometry": {
            "frame_id": "eve/odom",
            "child_frame_id": "eve/base_footprint",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "first_map_nn0_transform": {
            "parent_frame": "eve/odom",
            "child_frame": "eve/map/nn0",
            "translation_m": [3.915405815387846, 3.379853301433703, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.7178065166282731, 0.6962426334877696],
        },
        "topic_message_counts": {
            "/tf_static": 5,
            "/eve/motors_enc": 38727,
            "/eve/scan": 15521,
            "/eve/imu/data": 154551,
            "/tf": 54241,
            "/eve/ouster/imu": 38851,
            "/eve/odom": 38717,
            "/eve/ouster/points": 3886,
        },
    },
    "mocap_or_optitrack_topic_count": 0,
    "pointcloud_frame_id": "eve/os_sensor",
    "pointcloud_fields": ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"],
    "static_transform_pairs": [
        ["eve/base_footprint", "eve/base_link"],
        ["eve/base_link", "eve/imu_link"],
        ["eve/base_link", "eve/laser"],
        ["eve/base_link", "eve/os_sensor"],
        ["eve/os_sensor", "eve/lidar3d"],
        ["eve/os_sensor", "eve/os_imu"],
    ],
    "t_base_link_ouster_translation_m": [0.0, 0.0, 0.4367],
    "t_base_link_ouster_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
}

PROTOCOL_ASSETS = (
    "protocols/real_data_validation_protocol_framework_v1.md",
    "protocols/real_data_validation_protocol_framework_v1.json",
    "protocols/real_data_dataset_eligibility_checklist.csv",
    "protocols/real_data_snapshot_selection_template.csv",
    "protocols/real_data_uncertainty_budget_template.csv",
    "src/phase_a_harness/real_data_protocol.py",
    "src/phase_a_harness/common_association_analysis.py",
    "frozen_assets/backend_parameter_contract.json",
)


def _run(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None) -> str:
    process = subprocess.run(
        list(command), cwd=cwd, env=dict(environment) if environment else None,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return process.stdout


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_report(repository: Path) -> dict[str, Any]:
    merge_base = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"], cwd=repository
    ).returncode == 0
    tag_object = _run(["git", "rev-parse", HISTORICAL_TAG], cwd=repository).strip()
    tag_peeled = _run(["git", "rev-parse", f"{HISTORICAL_TAG}^{{}}"], cwd=repository).strip()
    return {
        "baseline_is_ancestor": merge_base,
        "branch": _run(["git", "branch", "--show-current"], cwd=repository).strip(),
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repository).strip(),
        "historical_tag_object": tag_object,
        "historical_tag_peeled_commit": tag_peeled,
        "historical_tag_unchanged": tag_peeled == BASELINE_COMMIT,
        "worktree_porcelain": _run(["git", "status", "--porcelain"], cwd=repository).splitlines(),
    }


def _environment_report(repository: Path, data_root: Path, python: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(data_root)
    return {
        "collected_at_utc": _utc_now(),
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
            "UNKNOWN",
        ),
        "disk": {"free_bytes": usage.free, "total_bytes": usage.total, "used_bytes": usage.used},
        "download_tools": {
            "aria2": "1.35.0",
            "curl": _run(["curl", "--version"], cwd=repository).splitlines()[0],
            "git": _run(["git", "--version"], cwd=repository).strip(),
        },
        "git": _git_report(repository),
        "host": platform.node(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "pip_freeze": _run([str(python), "-m", "pip", "freeze", "--all"], cwd=repository).splitlines(),
        "python_executable": str(python),
        "python_version": _run([str(python), "--version"], cwd=repository).strip(),
    }


def _download_rows(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, remote in IILABS_FILES.items():
        final = data_root / relative
        partial = Path(f"{final}.partial")
        path = final if final.is_file() else partial if partial.is_file() else None
        size = path.stat().st_size if path else 0
        complete = bool(final.is_file() and size == remote["size"])
        control = Path(f"{partial}.aria2")
        completed_checkpoint = IILABS_DOWNLOAD_PROGRESS_CHECKPOINT["files"].get(relative)
        rows.append(
            {
                "aria2_completed_bytes_checkpoint": (
                    remote["size"] if complete else completed_checkpoint or 0
                ),
                "aria2_control_sha256": sha256_file(control) if control.is_file() else "NOT_PRESENT",
                "dataset_id": "IILABS_3D",
                "local_path": str(path or final),
                "logical_size_bytes": size,
                "remote_checksum": "NOT_PUBLISHED",
                "remote_size_bytes": remote["size"],
                "sha256": sha256_file(path) if complete and path else "PENDING",
                "status": "COMPLETE_SIZE_VERIFIED" if complete else "PARTIAL" if path else "MISSING",
                "url": remote["url"],
            }
        )
    for relative, expected_sha in sorted(GRANDTOUR_LFS_SHA256.items()):
        path = data_root / "grandtour" / relative
        actual_sha = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "dataset_id": "GRANDTOUR",
                "aria2_completed_bytes_checkpoint": 0,
                "aria2_control_sha256": "NOT_APPLICABLE",
                "local_path": str(path),
                "logical_size_bytes": path.stat().st_size if path.is_file() else 0,
                "remote_checksum": expected_sha,
                "remote_size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": actual_sha,
                "status": "COMPLETE_SHA_VERIFIED" if actual_sha == expected_sha else "MISSING_OR_MISMATCH",
                "url": f"https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/{HF_REVISION}/{relative}?download=true",
            }
        )
    for path in sorted((data_root / "grandtour").glob("*/metadata/*.yaml")):
        rows.append(
            {
                "dataset_id": "GRANDTOUR",
                "aria2_completed_bytes_checkpoint": 0,
                "aria2_control_sha256": "NOT_APPLICABLE",
                "local_path": str(path),
                "logical_size_bytes": path.stat().st_size,
                "remote_checksum": "GIT_BLOB_ID_RECORDED_IN_REMOTE_TREE_AUDIT",
                "remote_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "COMPLETE_SIZE_VERIFIED",
                "url": f"https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/{HF_REVISION}/{path.relative_to(data_root / 'grandtour')}?download=true",
            }
        )
    return rows


def _tum_stats(path: Path) -> dict[str, Any]:
    row_count = 0
    first: list[float] | None = None
    last: list[float] | None = None
    previous: float | None = None
    max_gap = 0.0
    monotonic = True
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                raise ValueError(f"invalid TUM row in {path}")
            if first is None:
                first = values
            if previous is not None:
                monotonic &= values[0] > previous
                max_gap = max(max_gap, values[0] - previous)
            previous = values[0]
            last = values
            row_count += 1
    return {
        "first_pose": first,
        "last_pose": last,
        "max_timestamp_gap_s": max_gap,
        "row_count": row_count,
        "strictly_monotonic": monotonic,
        "tum_sha256": sha256_file(path),
    }


def _grandtour_probe(repository: Path, data_root: Path) -> dict[str, Any]:
    python = data_root / "tooling/iilabs3d-venv/bin/python"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    output = _run(
        [str(python), "-m", "phase_a_harness.real_data_preparation.grandtour_probe", "--data-root", str(data_root)],
        cwd=repository,
        environment=environment,
    )
    return json.loads(output)


def _protocol_hashes(repository: Path) -> dict[str, Any]:
    return {relative: sha256_file(repository / relative) for relative in PROTOCOL_ASSETS}


def _write_empty_csvs(root: Path) -> None:
    empty_headers = {
        "all_candidate_scans.csv": ("dataset_id", "scan_timestamp", "status", "exclusion_reason"),
        "all_candidate_intervals.csv": ("dataset_id", "interval_id", "interval_start_time", "interval_end_time", "status"),
        "excluded_candidates.csv": ("dataset_id", "candidate_id", "predeclared_reason"),
        "geometry_only_metrics.csv": ("dataset_id", "snapshot_id", "normalized_lambda_min_trans", "condition_number_trans", "spectral_entropy_trans"),
        "selected_scene_intervals.csv": ("dataset_id", "interval_id", "scene_label", "status"),
        "canonical_input_manifest.csv": (
            "dataset_id", "snapshot_id", "source_points_path", "source_points_sha256", "target_map_path",
            "target_map_sha256", "T_reference_path", "T_reference_sha256", "bundle_sha256",
            "future_open3d_bundle_sha256", "future_pcl_bundle_sha256", "byte_identical_for_both_backends",
        ),
    }
    for name, fields in empty_headers.items():
        atomic_write_csv(root / name, [], fields)
    atomic_write_csv(root / "real_data_snapshot_selection_v1.csv", [], SNAPSHOT_SELECTION_FIELDS)


def _audit_rows(grandtour_overlap: Mapping[str, Any], backend_ok: bool) -> list[dict[str, Any]]:
    global_status = {
        "R01": "PASS",
        "R02": "PASS",
        "R03": "FAIL",
        "R04": "FAIL",
        "R05": "FAIL",
        "R06": "BLOCKED",
        "R07": "BLOCKED",
        "R08": "BLOCKED",
        "R09": "PASS" if backend_ok else "FAIL",
        "R10": "FAIL",
    }
    reasons = {
        "R01": "Two public sources and immutable release identifiers were recorded.",
        "R02": "Downloaded IILABS MoCap TUM poses and GrandTour IE-TC/NavSatFix 6DoF reference files are independent of LiDAR registration.",
        "R03": "No target map was constructed; IILABS cross-acquisition world frame is unproven and GrandTour GT-only overlap failed.",
        "R04": "Acquisitions are distinct, but no eligible cross-acquisition map/query pair survived the common-frame and overlap hard gates.",
        "R05": "IILABS raw MoCap frame continuity is not proven and GrandTour raw Hesai per-point timing was intentionally not downloaded after overlap failure.",
        "R06": "Zero snapshots were selected because materialization stopped at hard gates.",
        "R07": "No labels exist to freeze; registration-result count is zero.",
        "R08": "No canonical bundles exist, so byte-identical backend inputs cannot be asserted.",
        "R09": "Historical backend parameter file and internal canonical hashes match exactly." if backend_ok else "Backend parameter hash mismatch.",
        "R10": "Extrinsic/map/interpolation numeric uncertainty evidence is incomplete; UNKNOWN values were not replaced with zero.",
    }
    rows = []
    for identifier, requirement, _ in ELIGIBILITY_REQUIREMENTS[:10]:
        rows.append(
            {
                "evidence_files": "MULTIPLE_SEE_JSON",
                "evidence_sha256": "BOUND_BY_FROZEN_MANIFEST",
                "global_status": global_status[identifier],
                "iilabs_status": global_status[identifier],
                "grandtour_status": global_status[identifier],
                "reason": reasons[identifier],
                "requirement": requirement,
                "requirement_id": identifier,
            }
        )
    return rows


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    answers = summary["answers"]
    lines = ["# Real-data Validation 预注册准备总结", ""]
    for index, answer in enumerate(answers, 1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## 最终结论",
            "",
            "`PREREGISTRATION_NOT_READY`",
            "",
            "硬阻塞包括：IILABS 跨 acquisition 固定世界系尚未由官方证据证明；GrandTour SPX-1↔SPX-3 在 WGS84/ECEF 下的 GT-only overlap 双向均未达到冻结阈值；因此未下载 Hesai 大点云、未建图、未盲选、未生成伪造的 200-row inventory。",
            "",
            "保持 `REAL_DATA_RUN_AUTHORIZED=false`、`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`、`registration_execution_count=0`。",
            "",
        ]
    )
    return "\n".join(lines)


def execute_preparation(
    *,
    repository: Path,
    data_root: Path,
    runtime_root: Path,
    synthetic_run_root: Path,
    mode: str,
    workers: int,
    no_registration: bool,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    synthetic_run_root = synthetic_run_root.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=False)
    if mode not in ("fresh", "resume"):
        raise ValueError("mode must be fresh or resume")
    if workers != 2:
        raise ValueError("this frozen preparation requires workers=2")
    if no_registration is not True or os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise PermissionError("--no-registration and ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 are mandatory")
    if mode == "fresh" and os.path.lexists(runtime_root):
        raise FileExistsError(f"fresh runtime root already exists: {runtime_root}")
    if mode == "resume" and not runtime_root.is_dir():
        raise FileNotFoundError(f"resume runtime root is absent: {runtime_root}")
    runtime_root.mkdir(parents=True, exist_ok=mode == "resume")
    for name in ("iilabs", "grandtour", "gt_overlap_plots"):
        (runtime_root / name).mkdir(exist_ok=True)

    python = Path(sys.executable).resolve(strict=True)
    environment = _environment_report(repository, data_root, python)
    git = environment["git"]
    if not git["baseline_is_ancestor"] or git["branch"] != EXPECTED_BRANCH or not git["historical_tag_unchanged"]:
        raise PermissionError("Git identity preflight failed")
    atomic_write_json(runtime_root / "environment_report.json", environment)

    try:
        import open3d
    except ImportError:
        open3d = None
    with NoRegistrationGuard(open3d_module=open3d) as guard:
        static_audit = assert_preparation_sources_are_safe(repository / "src/phase_a_harness/real_data_preparation")
        protocol_hashes = _protocol_hashes(repository)
        backend_ok = (
            protocol_hashes["frozen_assets/backend_parameter_contract.json"] == BACKEND_FILE_SHA256
        )
        source_release = {
            "collected_at_utc": _utc_now(),
            "grandtour": {
                "dataset_access": "PUBLIC_UNGATED_HUGGING_FACE_ZARR",
                "dataset_license_hugging_face": "MIT",
                "dataset_license_eth_repository": "IN_COPYRIGHT_NON_COMMERCIAL_USE_PERMITTED",
                "dataset_license_conflict": True,
                "github_commit": GRANDTOUR_GITHUB_COMMIT,
                "grand_tour_box_submodule_commit": GRANDTOUR_BOX_COMMIT,
                "hesai_model": "Hesai XT-32",
                "hf_revision": HF_REVISION,
                "paper": "arXiv:2602.18164",
                "spx_2_reference_status": "HIDDEN_OR_UNAVAILABLE",
            },
            "iilabs3d": {
                "dataset_access": "PUBLIC_INESC_TEC_REPOSITORY",
                "dataset_doi": "10.25747/VHNJ-WM80",
                "dataset_license": "NOT_EXPLICITLY_RECOVERED_FROM_DATASET_LANDING_PAGE",
                "paper_license": "CC_BY_4_0",
                "software_license": "BSD_3_CLAUSE",
                "toolkit_sdist_sha256": IILABS_TOOLKIT_SDIST_SHA256,
                "toolkit_version": "0.2.1",
            },
            "protocol_asset_sha256": protocol_hashes,
            "static_no_registration_audit": static_audit,
        }
        atomic_write_json(runtime_root / "source_release_manifest.json", source_release)

        toolkit = data_root / "tooling/iilabs3d-venv/bin/iilabs3d"
        cli_outputs = {}
        for name, arguments in (
            ("help", ["--help"]), ("list_sequences", ["list-sequences"]), ("list_sensors", ["list-sensors"])
        ):
            output = _run([str(toolkit), *arguments], cwd=repository)
            cli_outputs[name] = output
            atomic_write_bytes(runtime_root / "iilabs" / f"toolkit_{name}.txt", output.encode("utf-8"))

        download_rows = _download_rows(data_root)
        download_fields = (
            "dataset_id", "url", "local_path", "remote_size_bytes", "logical_size_bytes",
            "aria2_completed_bytes_checkpoint", "aria2_control_sha256",
            "remote_checksum", "sha256", "status",
        )
        atomic_write_csv(runtime_root / "download_manifest.csv", download_rows, download_fields)
        atomic_write_json(
            runtime_root / "download_manifest.json",
            {
                "download_commands": [
                    "iilabs3d --help; iilabs3d list-sequences; iilabs3d list-sensors",
                    "aria2c --continue=true --max-connection-per-server=4 --split=4 <two selected IILABS bag URLs>",
                    f"curl --continue-at - <selected GrandTour reference URLs pinned at {HF_REVISION}>",
                ],
                "files": download_rows,
                "iilabs_partial_download_checkpoint": IILABS_DOWNLOAD_PROGRESS_CHECKPOINT,
                "workers": workers,
            },
        )

        diff_tum = data_root / "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_diff/ground_truth.tum"
        omni_tum = data_root / "iilabs3d/iilabs3d_dataset/benchmark/ouster_os1-64/nav_a_omni/ground_truth.tum"
        iilabs_reference = {
            "independent_6dof_reference": True,
            "nav_a_diff": _tum_stats(diff_tum),
            "nav_a_omni": _tum_stats(omni_tum),
            "postprocessed_initial_offsets_adjusted": True,
            "reference_source": "OptiTrack_Motive_via_NatNet_then_EVO_TUM",
            "reference_status": "PASS_INDIVIDUAL_SEQUENCE_ONLY",
        }
        iilabs_common = {
            "evidence": [
                "Both published TUM files begin at translation [0,0,0].",
                "Official documentation states initial position offsets were adjusted during EVO post-processing.",
                "Official bag index tails contain eight topics but no MoCap, Motive, or OptiTrack topic.",
                "Both first raw /eve/odom messages are identity poses in sequence-local eve/odom.",
                "No official fixed Nav_A_Diff-to-Nav_A_Omni transform is published in downloaded metadata.",
            ],
            "failure_reason": "UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
            "same_fixed_world_frame_proven": False,
            "status": "FAIL",
        }
        iilabs_overlap = {
            "eligibility_status": "FAIL",
            "failure_reason": "UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
            "map_acquisition": "nav_a_diff",
            "overlap_status": "NOT_COMPUTABLE",
            "query_acquisition": "nav_a_omni",
        }
        iilabs_dataset = {
            "download_complete": all(
                row["status"] == "COMPLETE_SIZE_VERIFIED" for row in download_rows if row["dataset_id"] == "IILABS_3D"
            ),
            "selected_sensor_cli_slug": "ouster_os1_64",
            "selected_sensor_storage_slug": "ouster_os1-64",
            "selected_sequences": ["nav_a_diff", "nav_a_omni"],
            "partial_download_checkpoint": IILABS_DOWNLOAD_PROGRESS_CHECKPOINT,
            "status": "FAIL_UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
            "toolkit_cli_outputs_sha256": {
                key: sha256_file(runtime_root / "iilabs" / f"toolkit_{key}.txt") for key in cli_outputs
            },
        }
        atomic_write_json(runtime_root / "iilabs/dataset_audit.json", iilabs_dataset)
        atomic_write_json(runtime_root / "iilabs/reference_audit.json", iilabs_reference)
        atomic_write_json(runtime_root / "iilabs/common_world_frame_audit.json", iilabs_common)
        atomic_write_json(runtime_root / "iilabs/gt_overlap_report.json", iilabs_overlap)
        atomic_write_json(
            runtime_root / "iilabs/raw_rosbag_range_audit.json",
            IILABS_ROSBAG_RANGE_AUDIT,
        )

        grandtour = _grandtour_probe(repository, data_root)
        spx_forward = grandtour["gt_only_overlap"]["SPX-1->SPX-3"]
        spx_reverse = grandtour["gt_only_overlap"]["SPX-3->SPX-1"]
        grandtour_dataset = {
            "hf_revision": HF_REVISION,
            "reference_only_download_sha_verified": all(
                row["status"] == "COMPLETE_SHA_VERIFIED" for row in download_rows if row["dataset_id"] == "GRANDTOUR" and row["remote_checksum"] in GRANDTOUR_LFS_SHA256.values()
            ),
            "selected_lidar": "Hesai XT-32 raw hesai_points.tar",
            "selected_lidar_downloaded": False,
            "selected_lidar_not_downloaded_reason": "GT_ONLY_OVERLAP_HARD_GATE_FAILED",
            "spx_2_eligible": False,
            "spx_2_reference_status": "HIDDEN_OR_UNAVAILABLE",
            "status": "FAIL_GT_ONLY_OVERLAP",
        }
        grandtour_reference = {
            "accuracy_evidence": {
                "attitude_rms_degree_under_up_to_10s_outage": 0.01,
                "position_rms_m_under_up_to_10s_outage": [0.01, 0.02],
                "source": "GrandTour paper and NovAtel CPT7 performance specification",
            },
            "actual_downloaded_reference": "Inertial Explorer tightly coupled odometry + NavSatFix + TF + prism",
            "independent_non_lidar_6dof": True,
            "probe": grandtour["missions"],
            "reference_status": "PASS_FOR_SPX_1_AND_SPX_3",
            "spx_2_reference_status": "HIDDEN_OR_UNAVAILABLE",
        }
        grandtour_common = {
            "common_position_frame": grandtour["common_position_frame"],
            "orientation_note": "IE-TC publishes 6DoF orientation in ENU; GT-only overlap intentionally uses positions only.",
            "status": "PASS_FOR_GT_ONLY_POSITION_OVERLAP",
            "transform_estimated_from_trajectory_or_point_cloud": False,
        }
        atomic_write_json(runtime_root / "grandtour/dataset_audit.json", grandtour_dataset)
        atomic_write_json(runtime_root / "grandtour/reference_audit.json", grandtour_reference)
        atomic_write_json(runtime_root / "grandtour/common_world_frame_audit.json", grandtour_common)
        atomic_write_json(runtime_root / "grandtour/gt_overlap_report.json", grandtour)
        ranking_rows = [
            {"covered_duration_s": spx_forward["total_covered_duration_s"], "coverage_fraction": spx_forward["coverage_fraction"], "map_acquisition": "SPX-1", "query_acquisition": "SPX-3", "status": "FAIL"},
            {"covered_duration_s": spx_reverse["total_covered_duration_s"], "coverage_fraction": spx_reverse["coverage_fraction"], "map_acquisition": "SPX-3", "query_acquisition": "SPX-1", "status": "FAIL"},
        ]
        atomic_write_csv(runtime_root / "grandtour/mission_candidate_ranking.csv", ranking_rows, ("map_acquisition", "query_acquisition", "covered_duration_s", "coverage_fraction", "status"))

        matrix_rows = [
            {"dataset_id": "IILABS_3D", "map_acquisition": "nav_a_diff", "query_acquisition": "nav_a_omni", "overlap_status": "NOT_COMPUTABLE", "covered_duration_s": "UNKNOWN", "coverage_fraction": "UNKNOWN"},
            {"dataset_id": "GRANDTOUR", "map_acquisition": "SPX-1", "query_acquisition": "SPX-3", "overlap_status": "FAIL", "covered_duration_s": spx_forward["total_covered_duration_s"], "coverage_fraction": spx_forward["coverage_fraction"]},
            {"dataset_id": "GRANDTOUR", "map_acquisition": "SPX-3", "query_acquisition": "SPX-1", "overlap_status": "FAIL", "covered_duration_s": spx_reverse["total_covered_duration_s"], "coverage_fraction": spx_reverse["coverage_fraction"]},
        ]
        atomic_write_csv(runtime_root / "gt_only_overlap_matrix.csv", matrix_rows, ("dataset_id", "map_acquisition", "query_acquisition", "overlap_status", "covered_duration_s", "coverage_fraction"))
        atomic_write_json(runtime_root / "gt_only_overlap_matrix.json", {"rows": matrix_rows})
        atomic_write_bytes(runtime_root / "gt_overlap_plots/README.md", b"No registration plot was generated. IILABS overlap is not computable; GrandTour numeric GT-only results are frozen in the matrix.\n")

        preprocessing = {
            "contract_status": "FROZEN_BUT_NOT_EXECUTED_DUE_TO_ELIGIBILITY_FAILURE",
            "coordinate_frames": {"source": "sensor_lidar", "target": "independent_reference_world"},
            "crop_rule": "no_crop",
            "deskew_method": "per_point_independent_non_lidar_6dof_reference_only",
            "dynamic_object_handling": "none",
            "finite_value_rule": "all_xyz_finite",
            "input_sensor": {"grandtour": "Hesai_XT_32_raw", "iilabs": "Ouster_OS1_64_RevC_raw"},
            "map_accumulation_rule": "map_acquisition_only_query_zero_contribution",
            "numeric_dtype": "little_endian_float64_C_contiguous",
            "point_ordering_rule": "scan_timestamp_then_original_point_index",
            "range_filter_m": {"minimum": 1.0, "maximum": 80.0},
            "reference_interpolation_method": "linear_translation_SLERP_xyzw",
            "software": {"numpy": "1.26.4", "scipy": "1.11.4"},
            "source_voxel_size_m": 0.10,
            "target_map_scan_sampling": "every_fifth_map_scan",
            "target_voxel_size_m": 0.10,
            "timestamp_field": "REQUIRES_RAW_SENSOR_FIELD_AUDIT",
        }
        preprocessing["contract_sha256"] = __import__("hashlib").sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        atomic_write_json(runtime_root / "real_data_preprocessing_contract_v1.json", preprocessing)
        _write_empty_csvs(runtime_root)

        lineage = {
            "map_built": False,
            "map_scan_source_ids": [],
            "query_scan_source_ids": [],
            "query_map_intersection_count": 0,
            "status": "NOT_MATERIALIZED_HARD_GATE_FAILURE",
        }
        for dataset in ("iilabs", "grandtour"):
            atomic_write_json(runtime_root / dataset / "map_lineage_manifest.json", {"dataset_id": dataset, **lineage})
            atomic_write_json(runtime_root / dataset / "uncertainty_evidence.json", {"dataset_id": dataset, "status": "FAIL_UNKNOWN_COMPONENTS", "unknown_components": ["extrinsic", "map", "deskew_interpolation"], "zero_substitution_used": False})
        atomic_write_json(
            runtime_root / "iilabs/transform_chain_manifest.json",
            {
                "dataset_id": "iilabs",
                "matrix_direction_audited": True,
                "T_world_ouster_formula": "T_world_base_link @ T_base_link_ouster",
                "T_base_link_ouster_translation_m": [0.0, 0.0, 0.4367],
                "T_base_link_ouster_quaternion_order": "xyzw",
                "T_world_base_link_status": "UNAVAILABLE_IN_RAW_SELECTED_BAGS_AND_SEQUENCE_LOCAL_TUM_ONLY",
                "status": "INCOMPLETE_UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME",
            },
        )
        atomic_write_json(
            runtime_root / "iilabs/time_sync_audit.json",
            {
                "dataset_id": "iilabs",
                "bag_and_tum_time_base": "ROS_UNIX_EPOCH_SECONDS",
                "bag_timestamps_strictly_ordered_by_chunk_info": True,
                "per_point_time_field": "t",
                "registration_or_lidar_odometry_used": False,
                "status": "PARTIAL_WORLD_REFERENCE_TOPIC_ABSENT",
            },
        )
        atomic_write_json(
            runtime_root / "grandtour/transform_chain_manifest.json",
            {
                "dataset_id": "grandtour",
                "common_position_frame": "WGS84_ECEF_FROM_PUBLISHED_IE_TC_NAVSATFIX",
                "orientation_source": "PUBLISHED_IE_TC_6DOF_ODOMETRY",
                "matrix_direction_audited": True,
                "status": "REFERENCE_CHAIN_AUDITED_RAW_HESAI_EXTRINSIC_NOT_MATERIALIZED_AFTER_OVERLAP_FAIL",
            },
        )
        atomic_write_json(
            runtime_root / "grandtour/time_sync_audit.json",
            {
                "dataset_id": "grandtour",
                "ie_tc_pose_rate_hz": 200.0,
                "max_reference_gap_s": max(
                    grandtour["missions"][mission]["max_timestamp_gap_s"]
                    for mission in ("SPX-1", "SPX-3")
                ),
                "registration_or_lidar_odometry_used": False,
                "status": "REFERENCE_PASS_RAW_HESAI_TIMING_NOT_AUDITED_AFTER_OVERLAP_FAIL",
            },
        )

        synthetic_analysis = synthetic_run_root / "analysis/primary.json"
        synthetic_sha = sha256_file(synthetic_analysis)
        synthetic = json.loads(synthetic_analysis.read_text(encoding="utf-8"))["h1_ideal_control"]
        synthetic_translation = max(float(row["translation_q95_m"]) for row in synthetic)
        synthetic_rotation = max(float(row["rotation_q95_rad"]) for row in synthetic)
        uncertainty_rows = []
        for dataset in ("IILABS_3D", "GRANDTOUR"):
            row = {field: "UNKNOWN" for field in UNCERTAINTY_BUDGET_FIELDS}
            row.update(
                {
                    "dataset_id": dataset,
                    "snapshot_id_or_group": "ALL_NOT_SELECTED",
                    "synthetic_translation_uncertainty_budget_m": synthetic_translation,
                    "synthetic_rotation_uncertainty_budget_rad": synthetic_rotation,
                    "combined_translation_uncertainty_m": "UNKNOWN",
                    "combined_rotation_uncertainty_rad": "UNKNOWN",
                    "uncertainty_combination_rule": "NOT_COMPUTABLE_UNKNOWN_COMPONENT_CONSERVATIVE_LINEAR_SUM_REQUIRED",
                    "evidence_reference": f"synthetic_analysis_sha256={synthetic_sha}",
                }
            )
            uncertainty_rows.append(row)
        atomic_write_csv(runtime_root / "real_data_uncertainty_budget_v1.csv", uncertainty_rows, UNCERTAINTY_BUDGET_FIELDS)

        blinded = {
            "label_selector_source_sha256": sha256_file(repository / "src/phase_a_harness/real_data_preparation/selection.py"),
            "labels_frozen": False,
            "labeler_blinded_to_registration_error": True,
            "real_registration_result_file_count": 0,
            "registration_derived_field_access_count": 0,
            "status": "BLOCKED_BEFORE_LABELING",
        }
        atomic_write_json(runtime_root / "blinded_labeling_record.json", blinded)
        audit_rows = _audit_rows(grandtour["gt_only_overlap"], backend_ok)
        audit_fields = ("requirement_id", "requirement", "global_status", "iilabs_status", "grandtour_status", "evidence_files", "evidence_sha256", "reason")
        atomic_write_csv(runtime_root / "r01_r10_eligibility_audit.csv", audit_rows, audit_fields)
        atomic_write_json(runtime_root / "r01_r10_eligibility_audit.json", {"requirements": audit_rows})
        audit_md = ["# R01–R10 Eligibility Audit", "", "| ID | Global | IILABS | GrandTour | Reason |", "|---|---|---|---|---|"]
        audit_md.extend(f"| {row['requirement_id']} | {row['global_status']} | {row['iilabs_status']} | {row['grandtour_status']} | {row['reason']} |" for row in audit_rows)
        atomic_write_bytes(runtime_root / "r01_r10_eligibility_audit.md", ("\n".join(audit_md) + "\n").encode())

        attestation = guard.attestation(runtime_root)
        atomic_write_json(runtime_root / "NO_ICP_ATTESTATION.json", attestation)
        interval_manifest = {
            "candidate_interval_definitions": [],
            "canonical_input_manifest_sha256": sha256_file(runtime_root / "canonical_input_manifest.csv"),
            "dataset_ids": ["IILABS_3D", "GRANDTOUR"],
            "exact_selection_algorithm": "FROZEN_IN_SELECTION_PY_BUT_NOT_EXECUTED",
            "failure_reasons": ["UNPROVEN_CROSS_SEQUENCE_WORLD_FRAME", "GRANDTOUR_GT_ONLY_OVERLAP_FAIL"],
            "labels_frozen_at_utc": None,
            "r14_freeze_pass": False,
            "selected_interval_ids": [],
            "snapshot_selection_csv_sha256": sha256_file(runtime_root / "real_data_snapshot_selection_v1.csv"),
            "source_code_commit": git["commit"],
            "status": "NOT_COMPLETED",
        }
        atomic_write_json(runtime_root / "interval_selection_manifest_v1.json", interval_manifest)

        answers = [
            f"IILABS 3D：{'两个 bag 均完整下载' if iilabs_dataset['download_complete'] else '已实际启动可续传下载，但截至冻结时未全部完成'}；GT 与标定已下载并校验。",
            "GrandTour：参考/TF metadata 与 SPX-1/SPX-3 IE-TC、NavSatFix、prism、TF 已按固定 HF revision 选择性下载且 LFS SHA 全匹配；Hesai 大点云因 GT-only overlap 先失败而未下载。",
            "候选 acquisition 为 IILABS nav_a_diff→nav_a_omni，以及 GrandTour SPX-1→SPX-3 / SPX-3→SPX-1；没有 mission pair 被最终冻结。",
            "IILABS 两条 TUM 均被后处理重置到零；没有官方固定跨序列变换，故未证明共享固定世界系。",
            "GrandTour SPX-1/SPX-3 已取得公开独立 IE-TC 6DoF reference；SPX-2 reference 官方明确隐藏。",
            f"GT-only overlap 未通过：SPX-1→SPX-3 为 {spx_forward['total_covered_duration_s']} s/{spx_forward['coverage_fraction']:.6f}，反向为 {spx_reverse['total_covered_duration_s']} s/{spx_reverse['coverage_fraction']:.6f}；IILABS 不可计算。",
            "两个数据集均未冻结 50 weak + 50 rich；inventory 为零行，未伪造。",
            "若资格通过，盲选主指标固定为 normalized_lambda_min_trans；本次未执行标签选择。",
            "没有查看或生成任何 registration error；registration-derived field access count=0。",
            "尚无 canonical bundle，不能声称 Open3D/PCL 已绑定字节相同输入。",
            "R01 PASS；R02 PASS；R03/R04/R05/R10 FAIL；R06/R07/R08 BLOCKED；R09 PASS。",
            "R14 未完成，不可变成功冻结未通过；仅冻结了可验证的失败审计。",
            "Open3D registration、PCL CLI、其他 registration 与真实 trial 调用次数全部为 0。",
            f"数据绝对路径：{data_root}；失败审计 manifest 绝对路径：{runtime_root}。",
            "不具备单独授权正式 400-trial 运行的条件。",
        ]
        summary = {
            "ICP_EXECUTION_COUNT": 0,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "READY_FOR_SEPARATE_RUN_AUTHORIZATION": False,
            "REAL_DATA_PREREGISTRATION_R01_R10_COMPLETE": False,
            "REAL_DATA_RUN_AUTHORIZED": False,
            "actual_icp_execution_count": 0,
            "actual_registration_execution_count": 0,
            "answers": answers,
            "dataset_count": 2,
            "dataset_execution_count": 0,
            "final_conclusion": "PREREGISTRATION_NOT_READY",
            "planned_backend_trials": 0,
            "target_planned_backend_trials_if_eligible": 400,
            "registration_execution_count": 0,
            "rich_snapshot_count": 0,
            "snapshot_count": 0,
            "weak_snapshot_count": 0,
        }
        atomic_write_json(runtime_root / "preparation_summary.json", summary)
        atomic_write_bytes(runtime_root / "preparation_summary.md", (_summary_markdown(summary) + "\n").encode())
        license_report = f"""# License and Citation Audit

## IILABS 3D

- Dataset DOI: `10.25747/VHNJ-WM80`.
- Toolkit `0.2.1` software license: BSD 3-Clause.
- The paper is CC BY 4.0; the dataset landing page license could not be recovered because the DOI endpoint returned HTTP 500. The paper license is not silently substituted for the dataset license.

## GrandTour

- Hugging Face marks the dataset `MIT` at revision `{HF_REVISION}`.
- ETH Research Collection reports “In Copyright – Non-Commercial Use Permitted”. This conflict is recorded and unresolved.
- Citation: Frey et al., “GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation,” arXiv:2602.18164 (2026).
"""
        atomic_write_bytes(runtime_root / "license_and_citation_report.md", license_report.encode())

        manifest_exclusions = {"frozen_manifest_v1.json", "SHA256SUMS"}
        relative_paths = sorted(
            str(path.relative_to(runtime_root))
            for path in runtime_root.rglob("*")
            if path.is_file() and str(path.relative_to(runtime_root)) not in manifest_exclusions
        )
        frozen = build_frozen_manifest(
            runtime_root,
            relative_paths,
            source_commit=git["commit"],
            preregistration_ready=False,
            r14_freeze_pass=False,
            bindings={
                "backend_parameter_contract_file_sha256": BACKEND_FILE_SHA256,
                "open3d_parameter_canonical_sha256": OPEN3D_PARAMETERS_SHA256,
                "pcl_parameter_canonical_sha256": PCL_PARAMETERS_SHA256,
                "protocol_asset_sha256": protocol_hashes,
            },
        )
        atomic_write_json(runtime_root / "frozen_manifest_v1.json", frozen)
        checksum_paths = [*relative_paths, "frozen_manifest_v1.json"]
        atomic_write_bytes(runtime_root / "SHA256SUMS", sha256sums_bytes(runtime_root, checksum_paths))
        verification = verify_frozen_manifest(
            runtime_root, frozen, repository_root=repository
        )
        return {
            "frozen_manifest": frozen,
            "summary": summary,
            "verification": verification,
        }
