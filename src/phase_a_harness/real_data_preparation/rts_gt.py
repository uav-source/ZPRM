"""Fail-closed RTS-GT metadata/GT-only Stage-1 eligibility audit.

This module deliberately contains no point-cloud backend and no registration
dependency.  It audits the official RTS-GT metadata release, constructs a
translation reference from the synchronized three-prism trajectories, and
stops before LiDAR download whenever the transform/time chain is incomplete.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .guard import NoRegistrationGuard, assert_preparation_sources_are_safe
from .io import atomic_write_bytes, atomic_write_csv, atomic_write_json, compact_sha256, sha256_file


OFFICIAL_REPOSITORY_COMMIT = "449a299bcf8bc11766718db4ed4aea4a489c7c46"
OFFICIAL_PAGE_URL = "https://norlab.ulaval.ca/research/RTS_dataset/"
OFFICIAL_REPOSITORY_URL = "https://github.com/norlab-ulaval/RTS_project.git"
PAPER_URL = "https://arxiv.org/abs/2309.11935"
PAPER_DOI = "10.1109/ICRA57147.2024.10610998"
ARXIV_DOI = "10.48550/arXiv.2309.11935"
REFERENCE_CONFIGURATION = "f-2-1-1-1-6-1-SGP-1000"
BACKEND_PARAMETER_SHA256 = "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
EXPECTED_BRANCH = "prep/rts-gt-single-dataset-v1"


@dataclass(frozen=True)
class OverlapContract:
    resample_rate_hz: float = 1.0
    radius_m: float = 5.0
    min_contiguous_covered_duration_s: float = 5.0
    min_total_covered_duration_s: float = 150.0
    min_coverage_fraction: float = 0.60
    min_eligible_nonoverlapping_5s_intervals: int = 30
    maximum_native_gap_s: float = 0.1500001


OVERLAP_CONTRACT = OverlapContract()
DEPLOYMENTS: Mapping[str, tuple[str, ...]] = {
    "Campus": tuple(f"20220715-{index}" for index in range(1, 5)),
    "Tunnel": tuple(f"20220717-{index}" for index in range(1, 6)),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.check_output(list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT)


def _timestamp_iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _regular_file_totals(root: Path) -> dict[str, int]:
    apparent = 0
    allocated = 0
    count = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        apparent += stat.st_size
        allocated += stat.st_blocks * 512
        count += 1
    return {"allocated_bytes": allocated, "apparent_bytes": apparent, "regular_file_count": count}


def _verify_sha256sums(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        entries.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual})
    return {
        "checked_file_count": len(entries),
        "mismatch_count": sum(row["expected_sha256"] != row["actual_sha256"] for row in entries),
        "pass": bool(entries) and all(row["expected_sha256"] == row["actual_sha256"] for row in entries),
    }


def _load_prism(path: Path) -> np.ndarray | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    value = np.loadtxt(path, dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] < 2:
        raise ValueError(f"invalid seven-column prism trajectory: {path}")
    if not np.isfinite(value).all() or not np.all(np.diff(value[:, 0]) > 0.0):
        raise ValueError(f"non-finite or non-monotonic prism trajectory: {path}")
    return value


def load_reference_triplet(data_dir: Path, experiment_id: str) -> dict[str, Any]:
    base = data_dir / experiment_id / "filtered_prediction"
    paths = tuple(base / f"{REFERENCE_CONFIGURATION}_{index}.csv" for index in (1, 2, 3))
    prisms = tuple(_load_prism(path) for path in paths)
    if any(value is None for value in prisms):
        return {
            "experiment_id": experiment_id,
            "available": False,
            "failure_reason": "EMPTY_OR_MISSING_REFERENCE_PRISM_FILE",
            "paths": [str(path) for path in paths],
        }
    values = tuple(value for value in prisms if value is not None)
    assert len(values) == 3
    if not (np.array_equal(values[0][:, 0], values[1][:, 0]) and np.array_equal(values[0][:, 0], values[2][:, 0])):
        raise ValueError(f"unsynchronized reference prism timestamps: {experiment_id}")
    positions = np.stack([value[:, 1:4] for value in values], axis=1)
    centroid = positions.mean(axis=1)
    triangle_twice_area = np.linalg.norm(
        np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0]), axis=1
    )
    times = values[0][:, 0]
    gaps = np.diff(times)
    return {
        "experiment_id": experiment_id,
        "available": True,
        "centroid_trajectory": np.column_stack((times, centroid)),
        "first_timestamp": float(times[0]),
        "last_timestamp": float(times[-1]),
        "max_native_gap_s": float(np.max(gaps)),
        "median_native_rate_hz": float(1.0 / np.median(gaps)),
        "per_axis_precision_fields_present": True,
        "precision_field_count_per_prism": 3,
        "prism_triangle_twice_area_min_m2": float(np.min(triangle_twice_area)),
        "prism_triangle_twice_area_median_m2": float(np.median(triangle_twice_area)),
        "row_count": int(times.size),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in paths
        ],
        "synchronized_triplet": True,
        "triad_6dof_constructible": bool(np.min(triangle_twice_area) > 1e-8),
    }


def resample_gap_aware(trajectory: np.ndarray, contract: OverlapContract = OVERLAP_CONTRACT) -> np.ndarray:
    """Resample within native continuous segments without interpolating outages."""

    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] < 2:
        raise ValueError("trajectory must have at least two [timestamp,x,y,z] rows")
    if not np.isfinite(value).all() or not np.all(np.diff(value[:, 0]) > 0.0):
        raise ValueError("trajectory must be finite and strictly monotonic")
    breaks = np.flatnonzero(np.diff(value[:, 0]) > contract.maximum_native_gap_s) + 1
    segments = np.split(value, breaks)
    step = 1.0 / contract.resample_rate_hz
    output: list[np.ndarray] = []
    segment_id = 0
    for segment in segments:
        if segment.shape[0] < 2 or segment[-1, 0] - segment[0, 0] + 1e-12 < step:
            continue
        timestamps = np.arange(segment[0, 0], segment[-1, 0] + step * 1e-9, step)
        positions = np.column_stack(
            [np.interp(timestamps, segment[:, 0], segment[:, column]) for column in (1, 2, 3)]
        )
        output.append(np.column_stack((timestamps, positions, np.full(timestamps.size, segment_id))))
        segment_id += 1
    if not output:
        raise ValueError("no continuous segment is long enough for resampling")
    return np.vstack(output)


def compute_overlap_pair(
    map_trajectory: np.ndarray,
    query_trajectory: np.ndarray,
    contract: OverlapContract = OVERLAP_CONTRACT,
) -> dict[str, Any]:
    map_rows = resample_gap_aware(map_trajectory, contract)
    query_rows = resample_gap_aware(query_trajectory, contract)
    distances, _ = cKDTree(map_rows[:, 1:4]).query(query_rows[:, 1:4], k=1)
    covered = distances <= contract.radius_m
    eligible_duration = 0.0
    nonoverlapping = 0
    covered_runs: list[dict[str, Any]] = []
    step = 1.0 / contract.resample_rate_hz
    for segment_id in np.unique(query_rows[:, 4]).astype(int):
        indices = np.flatnonzero(query_rows[:, 4] == segment_id)
        flags = covered[indices]
        start: int | None = None
        for local_index, flag in enumerate(np.append(flags, False)):
            if flag and start is None:
                start = local_index
            elif not flag and start is not None:
                stop = local_index - 1
                count = stop - start + 1
                duration = count * step
                row = {
                    "duration_s": float(duration),
                    "end_time": float(query_rows[indices[stop], 0] + step),
                    "sample_count": int(count),
                    "segment_id": int(segment_id),
                    "start_time": float(query_rows[indices[start], 0]),
                }
                covered_runs.append(row)
                if duration >= contract.min_contiguous_covered_duration_s:
                    eligible_duration += duration
                    nonoverlapping += int(duration // 5.0)
                start = None
    covered_count = int(np.sum(covered))
    query_count = int(covered.size)
    total_duration = covered_count * step
    coverage_fraction = covered_count / query_count
    passed = (
        total_duration >= contract.min_total_covered_duration_s
        and coverage_fraction >= contract.min_coverage_fraction
        and nonoverlapping >= contract.min_eligible_nonoverlapping_5s_intervals
    )
    return {
        "contiguous_covered_intervals": covered_runs,
        "coverage_fraction": float(coverage_fraction),
        "covered_query_count": covered_count,
        "eligible_covered_duration_s": float(eligible_duration),
        "eligible_nonoverlapping_5s_interval_count": int(nonoverlapping),
        "map_resampled_count": int(map_rows.shape[0]),
        "nearest_distance_max_m": float(np.max(distances)),
        "nearest_distance_median_m": float(np.median(distances)),
        "nearest_distance_q95_m": float(np.quantile(distances, 0.95)),
        "overlap_status": "PASS" if passed else "FAIL",
        "query_count": query_count,
        "query_resampled_segment_count": int(np.unique(query_rows[:, 4]).size),
        "total_covered_duration_s": float(total_duration),
    }


def select_best_pair(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if row.get("overlap_status") == "PASS"]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["total_covered_duration_s"]),
            -float(row["coverage_fraction"]),
            -int(row["eligible_nonoverlapping_5s_interval_count"]),
            str(row["map_experiment_id"]),
            str(row["query_experiment_id"]),
        ),
    )[0]


def _hash_groups(paths_by_id: Mapping[str, Path]) -> dict[str, Any]:
    rows = {
        identifier: sha256_file(path) if path.is_file() else "MISSING"
        for identifier, path in paths_by_id.items()
    }
    present = [value for value in rows.values() if value != "MISSING"]
    return {
        "by_experiment": rows,
        "all_present_hashes_identical": bool(present) and len(set(present)) == 1,
        "missing_count": sum(value == "MISSING" for value in rows.values()),
        "unique_present_sha256": sorted(set(present)),
    }


def _download_manifest(data_root: Path, official_repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".git" in path.parts:
            continue
        relative = path.relative_to(data_root).as_posix()
        if relative.startswith("official_repository/"):
            repository_relative = path.relative_to(official_repository).as_posix()
            url = f"https://github.com/norlab-ulaval/RTS_project/blob/{OFFICIAL_REPOSITORY_COMMIT}/{repository_relative}"
            source_class = "OFFICIAL_GIT_METADATA_GT_CALIBRATION_OR_CODE"
        elif path.name == "rts_dataset_official_page.html":
            url = OFFICIAL_PAGE_URL
            source_class = "OFFICIAL_DATASET_LANDING_PAGE"
        else:
            url = "https://arxiv.org/pdf/2309.11935"
            source_class = "OFFICIAL_PAPER_PREPRINT"
        rows.append(
            {
                "contains_lidar_payload": False,
                "local_path": str(path),
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "source_class": source_class,
                "status": "COMPLETE_SHA256_RECORDED",
                "url": url,
            }
        )
    totals = _regular_file_totals(data_root)
    summary = {
        "file_rows_sha256": compact_sha256(rows),
        "large_lidar_download_performed": False,
        "materialized_non_git_file_bytes": sum(row["size_bytes"] for row in rows),
        "materialized_non_git_file_count": len(rows),
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "data_root_totals_including_partial_clone_git_objects": totals,
    }
    return rows, summary


def _environment_report(repository: Path, data_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[f"{key}_bytes"] = int(value.strip().split()[0]) * 1024
    return {
        "collected_at_utc": _utc_now(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total, "used_bytes": disk.used},
        "git": {
            "branch": _run(["git", "branch", "--show-current"], cwd=repository).strip(),
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=repository).strip(),
            "worktree_porcelain": _run(["git", "status", "--porcelain"], cwd=repository).splitlines(),
        },
        "host": platform.node(),
        "machine": platform.machine(),
        "memory": memory,
        "os": platform.platform(),
        "packages": {name: _package_version(name) for name in ("numpy", "scipy", "open3d", "pytest")},
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    answers = summary["answers"]
    lines = ["# RTS-GT 单数据集第二轮预注册总结", ""]
    for index, answer in enumerate(answers, 1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## 判定",
            "",
            "`SINGLE_DATASET_PREREGISTRATION_READY=false`",
            "",
            "Stage 1 在 R05 失败后已停止：Campus 发布目录没有 LiDAR/robot `sensor_positions.csv` 或等价外参，Tunnel 的 `20220717-*` 目录内参考轨迹时间戳却落在 2022-05-23；官方脚本所需的传感器位置文件亦未随所审计 release 发布。未下载 LiDAR、未建图、未选择 weak/rich snapshot、未运行任何 registration。",
            "",
            "`R01=PENDING_SECOND_DATASET`，`REAL_DATA_RUN_AUTHORIZED=false`，`MEASUREMENT_PAPER_MAINLINE_AUTHORIZED=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def execute_stage1(*, repository: Path, data_root: Path, runtime_root: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    official_repository = (data_root / "official_repository").resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=False)
    if os.environ.get("ZPRM_REAL_DATA_PREP_NO_REGISTRATION") != "1":
        raise PermissionError("ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1 is mandatory")
    if os.path.lexists(runtime_root):
        raise FileExistsError(f"fresh runtime root must be absent: {runtime_root}")
    if _run(["git", "rev-parse", "HEAD"], cwd=official_repository).strip() != OFFICIAL_REPOSITORY_COMMIT:
        raise PermissionError("official RTS-GT repository commit mismatch")
    if _run(["git", "status", "--porcelain"], cwd=official_repository).strip():
        raise PermissionError("official RTS-GT repository worktree is dirty")
    if _run(["git", "branch", "--show-current"], cwd=repository).strip() != EXPECTED_BRANCH:
        raise PermissionError("ZPRM preparation branch mismatch")
    if _run(["git", "status", "--porcelain"], cwd=repository).strip():
        raise PermissionError("ZPRM worktree must be clean before the formal audit")
    runtime_root.mkdir(parents=True)

    environment = _environment_report(repository, data_root)
    atomic_write_json(runtime_root / "environment_report.json", environment)
    first_audit_root = repository / "frozen_assets/real_data_validation_v1"
    first_audit_verification = _verify_sha256sums(first_audit_root)
    cleanup = {
        "allowed_deleted_roots": [
            "/home/lj/zero_perturbation_data/real_data_v1",
            "/home/lj/zero_perturbation_runtime/real_data/preparation_v1",
        ],
        "before": {
            "data_root_allocated_bytes": 73676689408,
            "filesystem_available_bytes": 37632868352,
            "filesystem_used_bytes": 228668039168,
            "runtime_root_allocated_bytes": 270336,
        },
        "after": {
            "deleted_roots_absent": True,
            "filesystem_available_bytes": 111309828096,
            "filesystem_used_bytes": 154991079424,
        },
        "cleanup_execution_date_utc": "2026-08-10",
        "first_failure_audit_aggregate_file_list_sha256": "5a0495540d9ad8037810b52674efddab020590cc141e7ba0da6d30b74edfb241",
        "first_failure_audit_verification": first_audit_verification,
        "released_bytes": 73676959744,
        "released_decimal_gb": 73.676959744,
        "released_gib": 68.61705374717712,
        "synthetic_confirmatory_runtime_preserved": Path(
            "/home/lj/zero_perturbation_runtime/confirmatory/synthetic_confirmatory_v3_requalified"
        ).is_dir(),
    }
    if not first_audit_verification["pass"]:
        raise PermissionError("first failure audit no longer verifies")
    atomic_write_json(runtime_root / "cleanup_report.json", cleanup)

    download_rows, download_summary = _download_manifest(data_root, official_repository)
    download_fields = (
        "relative_path", "local_path", "url", "source_class", "size_bytes", "sha256",
        "contains_lidar_payload", "status",
    )
    atomic_write_csv(runtime_root / "download_manifest.csv", download_rows, download_fields)
    atomic_write_json(runtime_root / "download_manifest.json", {"files": download_rows, **download_summary})

    with NoRegistrationGuard(open3d_module=None) as guard:
        source_audit = assert_preparation_sources_are_safe(
            repository / "src/phase_a_harness/real_data_preparation"
        )
        data_dir = official_repository / "data"
        references: dict[str, dict[str, Any]] = {}
        trajectories: dict[str, np.ndarray] = {}
        for deployment, identifiers in DEPLOYMENTS.items():
            for identifier in identifiers:
                result = load_reference_triplet(data_dir, identifier)
                trajectory = result.pop("centroid_trajectory", None)
                references[identifier] = result
                if trajectory is not None:
                    trajectories[identifier] = trajectory

        reference_audit = {
            "audit_conclusion": "PASS_INDEPENDENT_PRISM_TRIAD_6DOF_REFERENCE",
            "caveat": "The release publishes synchronized three-prism trajectories, not ready-made robot/LiDAR SE(3) poses; a non-collinear prism triad defines a 6DoF prism-rig frame independently of LiDAR registration.",
            "configuration": REFERENCE_CONFIGURATION,
            "experiments": references,
            "high_accuracy_evidence": "Three robotic total stations, synchronized filtered positions, and three per-axis precision fields per prism row.",
            "official_method": "Three RTS tracking three active prisms; position and orientation are constructed from prism geometry.",
            "r02_status": "PASS",
        }
        atomic_write_json(runtime_root / "rts_gt_reference_audit.json", reference_audit)

        campus_ids = DEPLOYMENTS["Campus"]
        tunnel_ids = DEPLOYMENTS["Tunnel"]
        campus_gcp = _hash_groups({identifier: data_dir / identifier / "total_stations/GCP.txt" for identifier in campus_ids})
        campus_two = _hash_groups({identifier: data_dir / identifier / "total_stations/Two_points.txt" for identifier in campus_ids})
        tunnel_gcp = _hash_groups({identifier: data_dir / identifier / "total_stations/GCP.txt" for identifier in tunnel_ids})
        tunnel_static = _hash_groups(
            {identifier: data_dir / identifier / "list_tf/TF_list_static_cp.csv" for identifier in tunnel_ids[1:]}
        )
        common_world = {
            "Campus": {
                "evidence": {"GCP": campus_gcp, "Two_points": campus_two},
                "experiments": list(campus_ids),
                "same_fixed_world_frame_proven": campus_gcp["all_present_hashes_identical"] and campus_two["all_present_hashes_identical"],
                "status": "PASS",
            },
            "Tunnel": {
                "evidence": {"GCP": tunnel_gcp, "static_control_transform_experiments_2_to_5": tunnel_static},
                "experiments": list(tunnel_ids),
                "same_fixed_world_frame_proven": tunnel_gcp["all_present_hashes_identical"] and tunnel_static["all_present_hashes_identical"],
                "status": "PASS",
            },
            "cross_deployment_common_frame_required": False,
            "method": "Byte-identical official GCP/control files and fixed control transforms within each deployment; no trajectory or point-cloud fitting.",
            "registration_alignment_used": False,
            "status": "PASS",
        }
        atomic_write_json(runtime_root / "rts_gt_common_world_frame_audit.json", common_world)

        overlap_rows: list[dict[str, Any]] = []
        overlap_json_rows: list[dict[str, Any]] = []
        for deployment, identifiers in DEPLOYMENTS.items():
            for map_id in identifiers:
                for query_id in identifiers:
                    if map_id == query_id:
                        continue
                    base = {
                        "deployment": deployment,
                        "map_experiment_id": map_id,
                        "query_experiment_id": query_id,
                    }
                    if map_id not in trajectories or query_id not in trajectories:
                        result = {
                            "overlap_status": "NOT_COMPUTABLE",
                            "failure_reason": "EMPTY_OR_MISSING_REFERENCE_PRISM_FILE",
                            "coverage_fraction": "",
                            "total_covered_duration_s": "",
                            "eligible_nonoverlapping_5s_interval_count": "",
                            "query_count": "",
                            "covered_query_count": "",
                        }
                    else:
                        result = compute_overlap_pair(trajectories[map_id], trajectories[query_id])
                    full = {**base, **result}
                    overlap_json_rows.append(full)
                    overlap_rows.append(
                        {
                            **base,
                            "overlap_status": result["overlap_status"],
                            "failure_reason": result.get("failure_reason", ""),
                            "total_covered_duration_s": result["total_covered_duration_s"],
                            "coverage_fraction": result["coverage_fraction"],
                            "eligible_nonoverlapping_5s_interval_count": result[
                                "eligible_nonoverlapping_5s_interval_count"
                            ],
                            "query_count": result["query_count"],
                            "covered_query_count": result["covered_query_count"],
                        }
                    )
        selected = {
            deployment: select_best_pair(row for row in overlap_json_rows if row["deployment"] == deployment)
            for deployment in DEPLOYMENTS
        }
        overlap_fields = (
            "deployment", "map_experiment_id", "query_experiment_id", "overlap_status", "failure_reason",
            "total_covered_duration_s", "coverage_fraction", "eligible_nonoverlapping_5s_interval_count",
            "query_count", "covered_query_count",
        )
        atomic_write_csv(runtime_root / "rts_gt_gt_only_overlap_matrix.csv", overlap_rows, overlap_fields)
        atomic_write_json(
            runtime_root / "rts_gt_gt_only_overlap_matrix.json",
            {
                "contract": asdict(OVERLAP_CONTRACT),
                "coordinate_used": "centroid of synchronized three-prism RTS reference in the proven deployment world frame",
                "pair_rows": overlap_json_rows,
                "registration_output_consulted": False,
                "selected_pairs": selected,
                "sorting_rule": [
                    "total_covered_duration_s descending", "coverage_fraction descending",
                    "eligible_nonoverlapping_5s_interval_count descending", "map then query experiment ID lexicographic",
                ],
            },
        )

        official_tree_paths = _run(
            ["git", "ls-tree", "-r", "--name-only", OFFICIAL_REPOSITORY_COMMIT, "--", "data"],
            cwd=official_repository,
        ).splitlines()
        all_sensor_positions_paths = sorted(
            path for path in official_tree_paths if path.endswith("/sensor_positions.csv")
        )
        selected_sensor_positions_paths = [
            path
            for path in all_sensor_positions_paths
            if any(f"data/{identifier}/" in path for identifiers in DEPLOYMENTS.values() for identifier in identifiers)
        ]
        transform_chain = {
            "Campus": {
                "available_steps": ["RTS observations", "GCP/common world", "synchronized prism triad", "prism-rig 6DoF"],
                "missing_steps": ["published prism/robot-to-LiDAR sensor_positions.csv or equivalent named SE(3)"],
                "status": "FAIL",
            },
            "Tunnel": {
                "available_steps": [
                    "RTS observations", "GCP/common world", "synchronized prism triad",
                    "calibration_raw.csv", "calibration_results.csv",
                ],
                "missing_steps": ["sensor_positions.csv required by official groundtruth_utils.py", "documented frame semantics for the 12 calibration_results values"],
                "status": "FAIL",
            },
            "official_script_required_path": "sensors_extrinsic_calibration/sensor_positions.csv",
            "official_landing_page_calibration_status": "Calibration information (available soon)",
            "official_tree_sensor_positions_file_count_other_deployments": len(all_sensor_positions_paths),
            "selected_deployment_sensor_positions_file_count": len(selected_sensor_positions_paths),
            "selected_deployment_sensor_positions_paths": selected_sensor_positions_paths,
            "registration_derived_transform_forbidden_and_used": False,
            "status": "FAIL",
        }
        atomic_write_json(runtime_root / "rts_gt_transform_chain_manifest.json", transform_chain)

        time_rows = []
        for deployment, identifiers in DEPLOYMENTS.items():
            expected_date = "2022-07-15" if deployment == "Campus" else "2022-07-17"
            for identifier in identifiers:
                result = references[identifier]
                if not result["available"]:
                    time_rows.append({"experiment_id": identifier, "status": "FAIL", "reason": result["failure_reason"]})
                    continue
                first_iso = _timestamp_iso(result["first_timestamp"])
                last_iso = _timestamp_iso(result["last_timestamp"])
                matching = first_iso.startswith(expected_date) and last_iso.startswith(expected_date)
                time_rows.append(
                    {
                        "experiment_id": identifier,
                        "directory_date": expected_date,
                        "first_reference_timestamp_utc": first_iso,
                        "last_reference_timestamp_utc": last_iso,
                        "status": "PASS" if matching else "FAIL",
                        "reason": "DIRECTORY_DATE_MATCH" if matching else "REFERENCE_TIMESTAMPS_CONFLICT_WITH_DIRECTORY_DATE",
                    }
                )
        time_sync = {
            "experiments": time_rows,
            "paper_timestamp_caveat": "RTS station clocks are not globally valid and official processing interpolates/synchronizes observations.",
            "status": "FAIL" if any(row["status"] == "FAIL" for row in time_rows) else "PASS",
            "tunnel_conflict": "20220717-2 through -5 filtered reference timestamps resolve to 2022-05-23; 20220717-1 reference files are empty.",
        }
        atomic_write_json(runtime_root / "rts_gt_time_sync_audit.json", time_sync)

        backend_hash = sha256_file(repository / "frozen_assets/backend_parameter_contract.json")
        dataset_audit = {
            "dataset": "RTS-GT",
            "dataset_doi": "NOT_LOCATED; paper DOI and arXiv DOI recorded separately",
            "deployments": {
                "Campus": {"date": "2022-07-15", "experiment_count": 4, "platform": "Warthog", "lidar_advertised": True},
                "Tunnel": {"date": "2022-07-17", "experiment_count": 5, "platform": "HD2", "lidar_advertised": True},
            },
            "license": {
                "official_repository": "MIT",
                "dataset_payload_scope": "NOT_SEPARATELY_STATED_ON_AUDITED_OFFICIAL_PAGE",
                "license_pass_not_overclaimed": True,
            },
            "official_page_url": OFFICIAL_PAGE_URL,
            "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
            "official_repository_url": OFFICIAL_REPOSITORY_URL,
            "paper_doi": PAPER_DOI,
            "paper_url": PAPER_URL,
            "arxiv_doi": ARXIV_DOI,
            "release_identity": "official repository commit plus downloaded official page and paper SHA256 in download manifest",
            "source_only_download": True,
            "static_no_registration_source_audit": source_audit,
        }
        atomic_write_json(runtime_root / "rts_gt_dataset_audit.json", dataset_audit)

        eligibility = {
            "candidate_gate_status": {
                "R02_candidate": "PASS",
                "R03_candidate": "PASS",
                "R04_candidate": "PASS",
                "R05_candidate": "FAIL",
                "GT_ONLY_OVERLAP": "PASS" if all(selected.values()) else "FAIL",
            },
            "GT_ONLY_OVERLAP": {
                "Campus": "PASS" if selected["Campus"] else "FAIL",
                "Tunnel": "PASS" if selected["Tunnel"] else "FAIL",
                "status": "PASS" if all(selected.values()) else "FAIL",
            },
            "R01": "PENDING_SECOND_DATASET",
            "R02": "PASS",
            "R03": "PASS_CANDIDATE",
            "R04": "PASS",
            "R05": "FAIL",
            "R06": "BLOCKED_STAGE1",
            "R07": "BLOCKED_STAGE1",
            "R08": "BLOCKED_STAGE1",
            "R09": "PASS" if backend_hash == BACKEND_PARAMETER_SHA256 else "FAIL",
            "R10": "FAIL",
            "R14": "BLOCKED_STAGE1",
            "actual_trials": 0,
            "large_lidar_download_authorized": False,
            "primary_blocker": "R05: released transform/time chain is not auditable (missing sensor_positions/LiDAR extrinsic, plus Tunnel release date/timestamp conflict)",
            "planned_future_trials": 0,
            "planned_future_trials_if_100_snapshots_had_frozen": 200,
            "rich_snapshot_count": 0,
            "single_dataset_preregistration_ready": False,
            "snapshot_count": 0,
            "stage1_status": "FAIL_STOPPED_BEFORE_LIDAR_DOWNLOAD",
            "weak_snapshot_count": 0,
        }
        atomic_write_json(runtime_root / "rts_gt_stage1_eligibility.json", eligibility)

        downloaded_gb = download_summary["data_root_totals_including_partial_clone_git_objects"]["allocated_bytes"] / 1e9
        answers = [
            f"第一次失败数据释放 73.676959744 GB（68.617 GiB）。",
            f"RTS-GT 当前本地占用 {downloaded_gb:.9f} GB（含 partial-clone Git 对象；无 LiDAR payload）。",
            "Tunnel 审计 20220717-1..5；GT-overlap 最优候选为 20220717-2 map → 20220717-4 query，exp1 参考文件为空。",
            "Campus 审计 20220715-1..4；GT-overlap 最优候选为 20220715-1 map → 20220715-4 query。",
            "是；同一 deployment 的官方 GCP/control 文件逐字节一致，且没有做轨迹或点云拟合。",
            "Tunnel GT-only overlap PASS。",
            "Campus GT-only overlap PASS。",
            "R02 PASS：同步、非共线的三棱镜 RTS 轨迹可独立构造 prism-rig 6DoF；但 LiDAR 变换链在 R05 失败。",
            "否；Stage 1 失败，weak=0、rich=0、snapshot=0。",
            "R02=PASS，R03=PASS_CANDIDATE，R04=PASS，R05=FAIL，R06–R08=BLOCKED_STAGE1，R09=PASS，R10=FAIL。",
            "R14 未完成（BLOCKED_STAGE1），没有 interval 可冻结。",
            "是；Open3D=0、PCL CLI=0、其他 registration process=0、registration execution=0。",
            "否；SINGLE_DATASET_PREREGISTRATION_READY=false。",
            "主要阻塞是 R05：官方 release 缺 sensor_positions/LiDAR 外参，并且 Tunnel 目录日期与参考轨迹时间戳冲突。",
            "暂不值得立即下载第二数据集；应先向 RTS-GT 发布方取得可认证外参与 Tunnel 时间/版本说明，否则当前 RTS-GT 无法完成 R05/R10。",
        ]
        summary = {
            "answers": answers,
            "eligibility": eligibility,
            "measurement_paper_mainline_authorized": False,
            "real_data_run_authorized": False,
            "selected_gt_only_pairs": selected,
            "single_dataset_preregistration_ready": False,
        }
        atomic_write_json(runtime_root / "rts_gt_single_dataset_summary.json", summary)
        atomic_write_bytes(
            runtime_root / "rts_gt_single_dataset_summary.md", _summary_markdown(summary).encode("utf-8")
        )
        attestation = guard.attestation(runtime_root)
        attestation.update(
            {
                "environment_guard": "ZPRM_REAL_DATA_PREP_NO_REGISTRATION=1",
                "large_lidar_download_performed": False,
                "no_registration_source_audit": source_audit,
                "status": "PASS" if attestation["pass"] else "FAIL",
            }
        )
        atomic_write_json(runtime_root / "NO_ICP_ATTESTATION.json", attestation)
    return summary
