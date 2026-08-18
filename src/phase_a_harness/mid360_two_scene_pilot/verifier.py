"""Independent, read-only verifier and final metadata freezer for the two-scene Pilot."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase_a_harness.mid360_pilot.bag_reader import sha256_file

from . import TWO_SCENE_FLAGS
from .pipeline import (
    DEFAULT_FROZEN_ROOT,
    DEFAULT_RUNTIME_ROOT,
    REQUIRED_BAG_SHA256,
    REPOSITORY_ROOT,
    SCENE_SPECS,
    read_csv,
    read_json,
    utc_now,
    write_json,
)
from .registration import write_runtime_sha256s


class TwoSceneVerificationError(RuntimeError):
    """Raised when any independent Pilot closure check fails."""


REQUIRED_FILES = (
    "input_bag_manifest.csv",
    "input_bag_manifest.json",
    "lidar_frame_audit.csv",
    "imu_staticity_audit.csv",
    "staticity_summary.json",
    "mid360_map_query_lineage.json",
    "mid360_two_scene_preprocessing_contract.json",
    "target_map_manifest.csv",
    "query_selection_frozen.csv",
    "query_selection_frozen.json",
    "geometry_only_metrics.csv",
    "geometry_scene_summary.json",
    "canonical_input_manifest.csv",
    "mid360_two_scene_debug_registration_authorization.json",
    "open3d_results.csv",
    "pcl_results.csv",
    "backend_execution_ledger.csv",
    "backend_input_identity_audit.csv",
    "mid360_two_scene_reassociation.csv",
    "mid360_two_scene_statistics.json",
    "mid360_two_scene_statistics.md",
    "mid360_two_scene_readiness.json",
    "mid360_two_scene_summary.json",
    "mid360_two_scene_summary.md",
    "SHA256SUMS",
)
EXPECTED_ROLE = {
    scene["map_name"]: (scene["scene_id"], "MAP") for scene in SCENE_SPECS
}
EXPECTED_ROLE.update(
    {scene["query_name"]: (scene["scene_id"], "QUERY") for scene in SCENE_SPECS}
)
EXPECTED_SCENE = {
    scene["scene_id"]: (scene["semantic_scene"], scene["scene_type"])
    for scene in SCENE_SPECS
}
GEOMETRY_FORBIDDEN_FRAGMENTS = (
    "t_est",
    "translation_",
    "rotation_",
    "displacement",
    "final_residual",
    "turnover",
    "fitness",
    "solver",
)


def _fail(message: str) -> None:
    raise TwoSceneVerificationError(message)


def _bool(value: str) -> bool:
    if value not in {"true", "false"}:
        _fail(f"invalid CSV boolean: {value!r}")
    return value == "true"


def _verify_sha256s(root: Path) -> dict[str, Any]:
    checksum_path = root / "SHA256SUMS"
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        if "  " not in line:
            _fail("malformed SHA256SUMS line")
        digest, relative = line.split("  ", 1)
        if relative in rows or len(digest) != 64:
            _fail("duplicate or malformed SHA256SUMS entry")
        rows[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(rows) != actual_files:
        missing = sorted(actual_files - set(rows))
        extra = sorted(set(rows) - actual_files)
        _fail(f"SHA256SUMS closure mismatch: missing={missing}, extra={extra}")
    bad = [
        relative
        for relative, expected in rows.items()
        if sha256_file(root / relative) != expected
    ]
    if bad:
        _fail(f"SHA256 mismatch: {bad}")
    return {"entry_count": len(rows), "all_match": True}


def _verify_flags(payload: Mapping[str, Any], label: str) -> None:
    expected = {
        "MID360_TWO_SCENE_PILOT": True,
        "INDEPENDENT_MAP_QUERY_ACQUISITION": True,
        "FORMAL_MEASUREMENT_RESULT": False,
        "PILOT_NONFORMAL_DO_NOT_CITE": True,
    }
    for key, value in expected.items():
        if payload.get(key) is not value:
            _fail(f"{label} flag mismatch: {key}")


def _verify_csv_flags(rows: Sequence[Mapping[str, str]], label: str) -> None:
    for index, row in enumerate(rows):
        expected = {
            "MID360_TWO_SCENE_PILOT": "true",
            "INDEPENDENT_MAP_QUERY_ACQUISITION": "true",
            "FORMAL_MEASUREMENT_RESULT": "false",
            "PILOT_NONFORMAL_DO_NOT_CITE": "true",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                _fail(f"{label}[{index}] flag mismatch: {key}")


def _load_npy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.dtype("<f8")
        or not array.flags.c_contiguous
        or array.ndim != 2
        or array.shape[1] != 3
        or array.shape[0] == 0
        or not np.all(np.isfinite(array))
    ):
        _fail(f"noncanonical NPY: {path}")
    return array


def _array_sha(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    import hashlib

    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _reflection_safe_rotation_angle(matrix: np.ndarray) -> float:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _verify_result_math(rows: Sequence[Mapping[str, str]], label: str) -> None:
    for row in rows:
        t0 = np.asarray(json.loads(row["T0"]), dtype=np.float64)
        estimate = np.asarray(json.loads(row["T_est"]), dtype=np.float64)
        delta_recorded = np.asarray(json.loads(row["Delta_T"]), dtype=np.float64)
        if not np.array_equal(t0, np.eye(4, dtype=np.float64)):
            _fail(f"{label} nonidentity T0: {row['snapshot_id']}")
        delta = np.linalg.inv(t0) @ estimate
        if not np.allclose(delta, delta_recorded, atol=1.0e-12, rtol=0.0):
            _fail(f"{label} Delta_T mismatch: {row['snapshot_id']}")
        translation = delta[:3, 3]
        recorded = np.asarray(
            [row["translation_x_m"], row["translation_y_m"], row["translation_z_m"]],
            dtype=np.float64,
        )
        if not np.allclose(translation, recorded, atol=1.0e-12, rtol=0.0):
            _fail(f"{label} translation vector mismatch: {row['snapshot_id']}")
        if not math.isclose(
            float(np.linalg.norm(translation)),
            float(row["translation_norm_m"]),
            abs_tol=1.0e-12,
            rel_tol=1.0e-10,
        ):
            _fail(f"{label} translation norm mismatch: {row['snapshot_id']}")
        angle = _reflection_safe_rotation_angle(delta[:3, :3])
        if not math.isclose(
            angle,
            float(row["rotation_angle_rad"]),
            abs_tol=1.0e-10,
            rel_tol=1.0e-8,
        ):
            _fail(f"{label} rotation metric mismatch: {row['snapshot_id']}")


def _protected_assets_unchanged(contract: Mapping[str, Any]) -> dict[str, Any]:
    bindings = contract.get("protected_tracked_file_sha256")
    if not isinstance(bindings, dict) or not bindings:
        _fail("protected tracked-file manifest missing")
    changed: list[str] = []
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            changed.append(relative)
    if changed:
        _fail(f"protected Boreas/Synthetic/Public-data assets changed: {changed}")
    return {"protected_file_count": len(bindings), "changed": []}


def verify_runtime(runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    runtime = runtime_root.expanduser().resolve(strict=True)
    missing = [name for name in REQUIRED_FILES if not (runtime / name).is_file()]
    if missing:
        _fail(f"required artifacts missing: {missing}")
    sha = _verify_sha256s(runtime)
    manifest = read_csv(runtime / "input_bag_manifest.csv")
    if len(manifest) != 4:
        _fail("bag manifest count is not 4")
    _verify_csv_flags(manifest, "input_bag_manifest")
    for row in manifest:
        path = Path(row["bag_path"]).resolve(strict=True)
        expected_role = EXPECTED_ROLE.get(path.name)
        if expected_role != (row["scene_id"], row["role"]):
            _fail(f"bag scene/role mapping mismatch: {path.name}")
        if EXPECTED_SCENE.get(row["scene_id"]) != (
            row["semantic_scene"],
            row["scene_type"],
        ):
            _fail(f"semantic scene label mismatch: {path.name}")
        expected_sha = REQUIRED_BAG_SHA256.get(path.name)
        if expected_sha is None or row["bag_sha256"] != expected_sha:
            _fail(f"bag manifest SHA binding mismatch: {path.name}")
        if sha256_file(path) != expected_sha:
            _fail(f"bag bytes changed: {path}")
    lineage = read_json(runtime / "mid360_map_query_lineage.json")
    _verify_flags(lineage, "lineage")
    if lineage.get("status") != "PASS" or lineage.get("same_bag_map_query") is not False:
        _fail("independent Map/Query lineage failed")
    for scene in SCENE_SPECS:
        row = lineage["scenes"].get(scene["scene_id"], {})
        if not (
            row.get("map_source") == scene["map_name"]
            and row.get("query_source") == scene["query_name"]
            and row.get("path_distinct") is True
            and row.get("sha256_distinct") is True
            and row.get("recording_intervals_nonoverlapping") is True
            and float(row.get("map_to_query_gap_seconds", -1.0)) > 0.0
        ):
            _fail(f"lineage scene binding failed: {scene['scene_id']}")
    staticity = read_json(runtime / "staticity_summary.json")
    _verify_flags(staticity, "staticity")
    if staticity.get("PILOT_REGISTRATION_BLOCKED_BY_STATICITY") is not False:
        _fail("registration should have been blocked by staticity")
    contract = read_json(runtime / "mid360_two_scene_preprocessing_contract.json")
    _verify_flags(contract, "preprocessing")
    if not (
        contract.get("status") == "PASS"
        and contract.get("target_voxel_size_m") == 0.05
        and contract.get("scene_specific_parameters") is False
        and contract.get("per_point_timestamp_order_assumed") is False
    ):
        _fail("preprocessing contract changed")
    protected = _protected_assets_unchanged(contract)
    target_rows = read_csv(runtime / "target_map_manifest.csv")
    if len(target_rows) != 2:
        _fail("target map count is not 2")
    _verify_csv_flags(target_rows, "target_map_manifest")
    for row in target_rows:
        target_path = Path(row["target_path"]).resolve(strict=True)
        target = _load_npy(target_path)
        metadata = read_json(target_path.parent / "metadata.json")
        if not (
            sha256_file(target_path) == row["target_npy_sha256"]
            and _array_sha(target) == row["target_array_sha256"]
            and int(row["voxelized_point_count"]) == target.shape[0]
            and row["target_registration_called"] == "false"
            and metadata.get("registration_called") is False
            and metadata.get("scan_matching_called") is False
            and metadata.get("odometry_called") is False
        ):
            _fail(f"target map construction/SHA failure: {row['scene_id']}")
    selection = read_csv(runtime / "query_selection_frozen.csv")
    if (
        len(selection) != 20
        or sum(row["scene_id"] == "R_TEST_01" for row in selection) != 10
        or sum(row["scene_id"] == "W_TEST_01" for row in selection) != 10
        or len({row["snapshot_id"] for row in selection}) != 20
    ):
        _fail("query selection is not exact 10 Rich + 10 Weak")
    _verify_csv_flags(selection, "query_selection")
    authorization = read_json(
        runtime / "mid360_two_scene_debug_registration_authorization.json"
    )
    _verify_flags(authorization, "authorization")
    if not (
        authorization.get("MAX_ALLOWED_TRIALS") == 40
        and authorization.get("authorized_trial_count") == 40
        and authorization.get("authorized_snapshot_count") == 20
        and authorization.get("query_selection_csv_sha256")
        == sha256_file(runtime / "query_selection_frozen.csv")
        and authorization.get("query_selection_json_sha256")
        == sha256_file(runtime / "query_selection_frozen.json")
        and authorization.get("backend_parameter_contract_sha256")
        == sha256_file(REPOSITORY_ROOT / "frozen_assets/backend_parameter_contract.json")
        and authorization.get("query_selection_frozen_before_backend") is True
        and authorization.get("formal_multisite_experiment_authorized") is False
    ):
        _fail("authorization binding failed")
    started = read_json(runtime / "backend_execution_started.json")
    if not (
        started.get("query_selection_csv_sha256")
        == sha256_file(runtime / "query_selection_frozen.csv")
        and started.get("geometry_only_metrics_sha256")
        == sha256_file(runtime / "geometry_only_metrics.csv")
        and started.get("pilot_backend_trial_count_before_start") == 0
    ):
        _fail("query/geometry freeze did not precede backend")
    with (runtime / "geometry_only_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        geometry_reader = csv.DictReader(stream)
        headers = [header.lower() for header in geometry_reader.fieldnames or []]
        geometry_rows = list(geometry_reader)
    if len(geometry_rows) != 20:
        _fail("geometry-only row count is not 20")
    if any(fragment in header for header in headers for fragment in GEOMETRY_FORBIDDEN_FRAGMENTS):
        _fail("registration-derived field escaped geometry-only artifact")
    geometry_summary = read_json(runtime / "geometry_scene_summary.json")
    _verify_flags(geometry_summary, "geometry_summary")
    summary_text = json.dumps(geometry_summary, sort_keys=True).lower()
    if any(fragment in summary_text for fragment in GEOMETRY_FORBIDDEN_FRAGMENTS):
        _fail("registration-derived content escaped geometry-only summary")
    canonical = read_csv(runtime / "canonical_input_manifest.csv")
    if len(canonical) != 20:
        _fail("canonical manifest row count is not 20")
    _verify_csv_flags(canonical, "canonical_input_manifest")
    selection_by_id = {row["snapshot_id"]: row for row in selection}
    canonical_by_id = {row["snapshot_id"]: row for row in canonical}
    if set(selection_by_id) != set(canonical_by_id):
        _fail("query selection/canonical snapshot ID mismatch")
    for row in canonical:
        source = _load_npy(Path(row["source_path"]).resolve(strict=True))
        target = _load_npy(Path(row["target_path"]).resolve(strict=True))
        if not (
            sha256_file(Path(row["source_path"])) == row["source_npy_sha256"]
            and _array_sha(source) == row["source_array_sha256"]
            and sha256_file(Path(row["target_path"])) == row["target_npy_sha256"]
            and _array_sha(target) == row["target_array_sha256"]
            and json.loads(row["T0"]) == np.eye(4).tolist()
            and EXPECTED_SCENE.get(row["scene_id"])
            == (row["semantic_scene"], row["scene_type"])
            and float(selection_by_id[row["snapshot_id"]]["query_timestamp"])
            == float(row["query_timestamp"])
            and int(selection_by_id[row["snapshot_id"]]["query_frame_index"])
            == int(row["query_frame_index"])
        ):
            _fail(f"canonical input drift: {row['snapshot_id']}")
    authorized_snapshots = {
        str(row["snapshot_id"]): row for row in authorization.get("snapshots", [])
    }
    if set(authorized_snapshots) != set(canonical_by_id):
        _fail("authorization/canonical snapshot ID mismatch")
    for snapshot_id, row in canonical_by_id.items():
        authorized = authorized_snapshots[snapshot_id]
        if not (
            float(authorized["query_timestamp"]) == float(row["query_timestamp"])
            and authorized["source_npy_sha256"] == row["source_npy_sha256"]
            and authorized["source_array_sha256"] == row["source_array_sha256"]
            and authorized["target_npy_sha256"] == row["target_npy_sha256"]
            and authorized["target_array_sha256"] == row["target_array_sha256"]
        ):
            _fail(f"authorization input binding mismatch: {snapshot_id}")
    open_rows = read_csv(runtime / "open3d_results.csv")
    pcl_rows = read_csv(runtime / "pcl_results.csv")
    ledger = read_csv(runtime / "backend_execution_ledger.csv")
    if not (
        len(open_rows) == 20
        and len(pcl_rows) == 20
        and len(ledger) == 40
        and len({row["trial_id"] for row in ledger}) == 40
        and {row["backend"] for row in open_rows} == {"open3d_point_to_plane"}
        and {row["backend"] for row in pcl_rows} == {"pcl_point_to_plane"}
        and {row["backend"] for row in ledger}
        == {"open3d_point_to_plane", "pcl_point_to_plane"}
    ):
        _fail("backend count or allowed-backend gate failed")
    _verify_csv_flags(open_rows, "open3d_results")
    _verify_csv_flags(pcl_rows, "pcl_results")
    for row in [*open_rows, *pcl_rows]:
        canonical_row = canonical_by_id.get(row["snapshot_id"])
        if canonical_row is None or not (
            row["scene_id"] == canonical_row["scene_id"]
            and row["semantic_scene"] == canonical_row["semantic_scene"]
            and row["scene_type"] == canonical_row["scene_type"]
            and float(row["query_timestamp"])
            == float(canonical_row["query_timestamp"])
            and row["source_sha256"] == canonical_row["source_npy_sha256"]
            and row["source_array_sha256"] == canonical_row["source_array_sha256"]
            and row["target_sha256"] == canonical_row["target_npy_sha256"]
            and row["target_array_sha256"] == canonical_row["target_array_sha256"]
        ):
            _fail(f"result/canonical binding mismatch: {row['trial_id']}")
    _verify_result_math(open_rows, "open3d")
    _verify_result_math(pcl_rows, "pcl")
    if not all(_bool(row["finite_result"]) for row in [*open_rows, *pcl_rows]):
        _fail("nonfinite backend result")
    identity = read_csv(runtime / "backend_input_identity_audit.csv")
    if len(identity) != 20 or not all(
        _bool(row["input_identity_pass"]) for row in identity
    ):
        _fail("Open3D/PCL input SHA identity failed")
    reassociation = read_csv(runtime / "mid360_two_scene_reassociation.csv")
    if len(reassociation) != 40 or not all(
        _bool(row["common_association_valid"]) for row in reassociation
    ):
        _fail("common reassociation closure failed")
    common_path = REPOSITORY_ROOT / "src/phase_a_harness/common_association_analysis.py"
    if contract["code_bindings"].get(
        "src/phase_a_harness/common_association_analysis.py"
    ) != sha256_file(common_path):
        _fail("common reassociation definition SHA changed")
    readiness = read_json(runtime / "mid360_two_scene_readiness.json")
    summary = read_json(runtime / "mid360_two_scene_summary.json")
    _verify_flags(readiness, "readiness")
    _verify_flags(summary, "summary")
    if not (
        readiness.get("MID360_TWO_SCENE_PILOT_READY") is True
        and summary.get("MID360_TWO_SCENE_PILOT_READY") is True
        and summary.get("formal_result_eligibility") == "NO"
        and summary.get("formal_multisite_experiment_started") is False
    ):
        _fail("readiness/formal boundary failure")
    return {
        "schema": "mid360_two_scene_independent_verification_v1",
        **TWO_SCENE_FLAGS,
        "INDEPENDENT_VERIFIER_PASS": True,
        "verified_at_utc": utc_now(),
        "checks": {
            "bag_sha_count": 4,
            "scene_role_mapping": "PASS",
            "independent_map_query_lineage": "PASS",
            "preprocessing_contract": "PASS",
            "target_maps_without_registration": 2,
            "target_map_sha": "PASS",
            "query_selection_prebackend_freeze": "PASS",
            "snapshot_count": 20,
            "rich_snapshot_count": 10,
            "weak_snapshot_count": 10,
            "canonical_arrays": "FINITE_LITTLE_ENDIAN_FLOAT64_C_CONTIGUOUS",
            "backend_input_byte_identity": "PASS",
            "identity_T0_count": 40,
            "trial_count": 40,
            "open3d_count": 20,
            "pcl_count": 20,
            "extra_backend_count": 0,
            "geometry_only_firewall": "PASS",
            "common_reassociation_definition": "PASS",
            "formal_flags": "PASS",
            "protected_assets": protected,
            "sha256_closure": sha,
            "result_math_recomputation": "PASS",
        },
        "status": "PASS",
    }


def _freeze_metadata(runtime: Path, frozen_root: Path) -> dict[str, Any]:
    if frozen_root.exists() and any(frozen_root.iterdir()):
        _fail(f"frozen output already exists; refusing overwrite: {frozen_root}")
    frozen_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    excluded: list[str] = []
    for source in sorted(path for path in runtime.rglob("*") if path.is_file()):
        relative = source.relative_to(runtime)
        relative_text = relative.as_posix()
        if source.name == "SHA256SUMS":
            destination = frozen_root / "RUNTIME_SHA256SUMS"
        elif source.suffix == ".npy" or relative_text == "imu_staticity_audit.csv":
            excluded.append(relative_text)
            continue
        else:
            destination = frozen_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "runtime_relative_path": relative_text,
                "frozen_relative_path": destination.relative_to(frozen_root).as_posix(),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    manifest = {
        "schema": "mid360_two_scene_frozen_asset_manifest_v1",
        **TWO_SCENE_FLAGS,
        "runtime_root": str(runtime),
        "large_arrays_committed_to_git": False,
        "raw_imu_audit_committed_to_git": False,
        "copied_file_count": len(copied),
        "copied_files": copied,
        "excluded_runtime_paths": excluded,
        "status": "PASS",
    }
    write_json(frozen_root / "frozen_asset_manifest.json", manifest)
    files = sorted(
        path
        for path in frozen_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (frozen_root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(frozen_root).as_posix()}\n"
            for path in files
        ),
        encoding="ascii",
    )
    return manifest


def finalize_verification_and_freeze(
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    frozen_root: Path = DEFAULT_FROZEN_ROOT,
) -> dict[str, Any]:
    runtime = runtime_root.expanduser().resolve(strict=True)
    frozen = frozen_root.expanduser().resolve()
    report_path = runtime / "independent_verification.json"
    if report_path.exists():
        _fail("independent verification already finalized; refusing overwrite")
    initial = verify_runtime(runtime)
    readiness = read_json(runtime / "mid360_two_scene_readiness.json")
    summary = read_json(runtime / "mid360_two_scene_summary.json")
    readiness["independent_verifier_pass"] = True
    readiness["independent_verification_status"] = "PASS"
    summary["independent_verifier_pass"] = True
    summary["independent_verification_status"] = "PASS"
    write_json(runtime / "mid360_two_scene_readiness.json", readiness)
    write_json(runtime / "mid360_two_scene_summary.json", summary)
    markdown_path = runtime / "mid360_two_scene_summary.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    if "- Independent verifier:" not in markdown:
        markdown = markdown.replace(
            "- MID360_TWO_SCENE_PILOT_READY:",
            "- Independent verifier: True\n- MID360_TWO_SCENE_PILOT_READY:",
        )
        markdown_path.write_text(markdown, encoding="utf-8")
    report = {
        **initial,
        "finalization_mode": "INDEPENDENT_VERIFY_THEN_FREEZE_METADATA",
        "runtime_sha256_regenerated_after_report": True,
        "frozen_root": str(frozen),
    }
    write_json(report_path, report)
    write_runtime_sha256s(runtime)
    final_runtime = verify_runtime(runtime)
    frozen_manifest = _freeze_metadata(runtime, frozen)
    frozen_sha = _verify_sha256s(frozen)
    return {
        **final_runtime,
        "frozen_asset_manifest": frozen_manifest,
        "frozen_sha256_closure": frozen_sha,
        "runtime_root": str(runtime),
        "frozen_root": str(frozen),
    }


__all__ = [
    "TwoSceneVerificationError",
    "finalize_verification_and_freeze",
    "verify_runtime",
]
