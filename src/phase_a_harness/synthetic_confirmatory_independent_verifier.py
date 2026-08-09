"""Independent H1--H6 recomputation for Synthetic Confirmatory v1.

This module intentionally does not import ``synthetic_confirmatory_analysis``.
All grouping, ranks, correlations, frozen-model predictions, and gate decisions
below are implemented a second time from record-level evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .phase_a_trial_result_schema import validate_phase_a_trial_result_strict


SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
CONDITIONS = ("IDEAL_MATCHED", "INDEPENDENT_NOISE_FREE", "FULL_NOISE")
BACKENDS = ("open3d_point_to_plane", "pcl_point_to_plane")
EPSILON = 1.0e-9
ABSOLUTE_TOLERANCE = 0.0
RELATIVE_TOLERANCE = 0.0
RAW_MANIFEST_SCHEMA = "synthetic_confirmatory_raw_result_manifest_v1"
_ASSOCIATION_LIMIT_M = 0.50
_PCA_NEIGHBORS = 50
_PCA_MIN_NEIGHBORS = 10
_PCA_CHUNK = 2048
_NORMAL_EPSILON = 1.0e-12
_COMMON_METRIC_FIELDS = (
    "initial_correspondence_count",
    "final_correspondence_count",
    "initial_valid_normal_correspondence_count",
    "final_valid_normal_correspondence_count",
    "correspondence_turnover",
    "accepted_source_turnover",
    "correspondence_count_change_ratio",
    "initial_residual_rmse",
    "initial_residual_median",
    "initial_residual_q95",
    "final_residual_rmse",
    "final_residual_median",
    "final_residual_q95",
    "residual_rmse_change",
    "median_normal_angle_change_deg",
    "q95_normal_angle_change_deg",
    "lambda_min_trans",
    "lambda_mid_trans",
    "lambda_max_trans",
    "normalized_lambda_min_trans",
    "normalized_lambda_mid_trans",
    "normalized_lambda_max_trans",
    "condition_number_trans",
    "spectral_entropy_trans",
    "initial_translation_gradient_norm",
)
_SNAPSHOT_FILES = frozenset(
    {"metadata.json", "source_points.npy", "target_points.npy", "reference_pose.npy"}
)
_SNAPSHOT_BUILDER_VERSION = "synthetic_confirmatory_snapshot_builder_v1"
_SNAPSHOT_LOCK_SCHEMA = "synthetic_confirmatory_snapshot_lock_v1"
_SNAPSHOT_METADATA_FIELDS = frozenset({
    "array_file_sha256",
    "condition",
    "confirmatory_rng_instantiation_count",
    "development_protocol_sha256",
    "dropout_parameters",
    "generator_sha256",
    "geometry_seed",
    "independent_sampling",
    "initial_pose",
    "measurement_seed",
    "metadata_payload_sha256",
    "noise_parameters",
    "planned_snapshot_id",
    "reference_pose_checksum",
    "repeat_index",
    "scene_variant",
    "snapshot_builder_contract_version",
    "snapshot_checksum",
    "snapshot_id",
    "source_checksum",
    "source_is_target_subset",
    "source_point_count",
    "target_checksum",
    "target_point_count",
})
_SNAPSHOT_LOCK_FIELDS = frozenset({
    "condition_snapshot_counts",
    "confirmatory_rng_instantiation_count",
    "planned_snapshot_count",
    "schema_version",
    "snapshot_builder_contract_version",
    "snapshot_lock_payload_sha256",
    "snapshots",
})
_SNAPSHOT_LOCK_ENTRY_FIELDS = frozenset({
    "condition",
    "confirmatory_rng_instantiation_count",
    "file_sha256",
    "geometry_seed",
    "measurement_seed",
    "metadata_payload_sha256",
    "reference_pose_checksum",
    "repeat_index",
    "scene_variant",
    "snapshot_checksum",
    "snapshot_id",
    "source_checksum",
    "target_checksum",
})
_CONDITION_METADATA = {
    "IDEAL_MATCHED": {
        "independent_sampling": False,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
    },
    "INDEPENDENT_NOISE_FREE": {
        "independent_sampling": True,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
    },
    "FULL_NOISE": {
        "independent_sampling": True,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.01,
        },
        "noise_parameters": {
            "map_noise_sigma_m": 0.001,
            "scan_noise_sigma_m": 0.003,
        },
    },
}
_INDEPENDENT_MANIFEST_BINDINGS = {
    "gitignore_runtime_outputs": ".gitignore",
    "core_contracts": "src/phase_a_harness/contracts.py",
    "scientific_protocol": "protocols/synthetic_confirmatory_protocol_v1.json",
    "scientific_protocol_document": "protocols/synthetic_confirmatory_protocol_v1.md",
    "gate_contract": "protocols/synthetic_confirmatory_gate_contract.json",
    "planned_snapshots": "protocols/synthetic_confirmatory_planned_snapshots.csv",
    "planned_trials": "protocols/synthetic_confirmatory_planned_trials.csv",
    "seed_provenance_audit": "protocols/confirmatory_seed_provenance_audit.json",
    "frozen_model": "frozen_assets/confirmatory_development_trained_models_v1.json",
    "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
    "pcl_cli": "bin/pcl_point_to_plane_cli",
    "trial_schema": "frozen_assets/trial_result_schema.json",
    "generator": "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py",
    "generator_wrapper": "src/phase_a_harness/phase_b_generator.py",
    "generator_development_protocol": "configs/zero_perturbation/development_v1.yaml",
    "generator_frozen_capture_init": "src/phase_a_harness/phase_b_generator_frozen/capture_range/__init__.py",
    "generator_frozen_capture_protocol": "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_protocol.py",
    "generator_frozen_capture_types": "src/phase_a_harness/phase_b_generator_frozen/capture_range/types.py",
    "generator_frozen_package_init": "src/phase_a_harness/phase_b_generator_frozen/__init__.py",
    "generator_frozen_zero_init": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/__init__.py",
    "generator_frozen_zero_protocol": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/protocol.py",
    "generator_frozen_zero_snapshot_builder": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/snapshot_builder.py",
    "generator_frozen_zero_types": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/types.py",
    "generator_frozen_phase_a_v1_2": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
    "generator_frozen_phase_a_protocol": "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_protocol.py",
    "snapshot_builder": "src/phase_a_harness/synthetic_confirmatory_snapshot_builder.py",
    "full_synthetic_snapshot_builder": "src/phase_a_harness/full_synthetic_snapshot_builder.py",
    "full_synthetic_development_protocol": "src/phase_a_harness/full_synthetic_development_protocol.py",
    "full_synthetic_development_runner": "src/phase_a_harness/full_synthetic_development_runner.py",
    "phase_b_snapshot_assets": "src/phase_a_harness/phase_b_snapshot_assets.py",
    "asset_verifier": "src/phase_a_harness/asset_verifier.py",
    "snapshot_reader": "src/phase_a_harness/snapshot_reader.py",
    "open3d_adapter": "src/phase_a_harness/open3d_backend.py",
    "pcl_adapter": "src/phase_a_harness/pcl_backend.py",
    "backend_metrics": "src/phase_a_harness/metrics.py",
    "backend_types": "src/phase_a_harness/types.py",
    "backend_phase_a_metrics": "src/phase_a_harness/backend_phase_a_metrics.py",
    "rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
    "phase_a_execution_chain": "src/phase_a_harness/phase_a_execution_chain_audit.py",
    "phase_a_execution_fixture": "src/phase_a_harness/phase_a_execution_chain_fixture.py",
    "fixture_source_access_monitor": "src/phase_a_harness/runner.py",
    "fixture_qualification_script": "scripts/run_fixture_qualification.py",
    "fixture_qualification": "src/phase_a_harness/fixture_qualification.py",
    "fixture_publication": "src/phase_a_harness/fixture_publication.py",
    "fixture_publication_artifact_verifier": "src/phase_a_harness/fixture_publication_artifact_verifier.py",
    "fixture_primary_analysis": "src/phase_a_harness/phase_a_stage1_analysis.py",
    "fixture_independent_verifier": "src/phase_a_harness/phase_a_stage1_independent_verifier.py",
    "fixture_plan": "frozen_assets/fixtures/fixture_plan.json",
    "fixture_snapshot_lock": "frozen_assets/fixtures/fixture_snapshot_lock.json",
    "fixture_backend_parameter_lock": "frozen_assets/fixtures/fixture_backend_parameter_lock.json",
    "full_synthetic_backend_execution": "src/phase_a_harness/full_synthetic_backend_execution.py",
    "full_synthetic_trial_result": "src/phase_a_harness/full_synthetic_trial_result.py",
    "phase_b_trial_result": "src/phase_a_harness/phase_b_trial_result.py",
    "phase_a_trial_result_schema": "src/phase_a_harness/phase_a_trial_result_schema.py",
    "trial_result_writer": "src/phase_a_harness/phase_a_trial_result_writer.py",
    "trial_resume": "src/phase_a_harness/phase_a_trial_resume.py",
    "phase_a_attempt_events": "src/phase_a_harness/phase_a_attempt_events.py",
    "common_association": "src/phase_a_harness/common_association_analysis.py",
    "confirmatory_protocol_builder": "src/phase_a_harness/confirmatory_protocol.py",
    "scientific_survival_models": "src/phase_a_harness/scientific_survival_models.py",
    "local_metric_models": "src/phase_a_harness/local_metric_models.py",
    "frozen_model_inference": "src/phase_a_harness/synthetic_confirmatory_models.py",
    "protocol_auditor": "src/phase_a_harness/synthetic_confirmatory_protocol.py",
    "analysis": "src/phase_a_harness/synthetic_confirmatory_analysis.py",
    "independent_verifier": "src/phase_a_harness/synthetic_confirmatory_independent_verifier.py",
    "publisher": "src/phase_a_harness/synthetic_confirmatory_publisher.py",
    "artifact_verifier": "src/phase_a_harness/synthetic_confirmatory_artifact_verifier.py",
    "runner": "src/phase_a_harness/synthetic_confirmatory_runner.py",
    "runner_script": "scripts/run_synthetic_confirmatory.py",
    "analysis_script": "scripts/analyze_synthetic_confirmatory.py",
    "independent_verifier_script": "scripts/verify_synthetic_confirmatory.py",
    "publisher_script": "scripts/publish_synthetic_confirmatory.py",
    "manifest_builder": "src/phase_a_harness/synthetic_confirmatory_manifest.py",
    "prerun_qualification": "src/phase_a_harness/synthetic_confirmatory_prerun.py",
    "prerun_qualification_script": "scripts/qualify_synthetic_confirmatory_prerun.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid independent JSON input: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"independent JSON root must be an object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("independent path escapes standalone repository")
    return candidate


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _raw_array_sha256(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise ValueError("independent raw checksum input is not C-contiguous")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _independent_validate_formal_manifest(
    manifest_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the sole authorized manifest without its implementation."""

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError("Synthetic Confirmatory formal manifest is missing")
    repository = manifest_file.parent.parent.resolve()
    expected_path = (
        repository / "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json"
    ).resolve()
    candidates = sorted(
        path.resolve()
        for path in (repository / "frozen_assets").glob(
            "synthetic_confirmatory_formal_manifest*.json"
        )
        if path.is_file()
    )
    if manifest_file != expected_path or candidates != [expected_path]:
        raise ValueError("independent verifier requires one exact formal manifest")
    manifest = _strict_object(manifest_file)
    alias_names = (
        "scientific_protocol",
        "scientific_protocol_document",
        "gate_contract",
        "planned_snapshots",
        "planned_trials",
        "seed_provenance_audit",
        "frozen_model",
    )
    expected_keys = {
        "backend_count", "bootstrap_seed", "bound_files",
        "formal_execution_authorized", "formal_output_dir", "formal_run_id",
        "formal_workers", "manifest_payload_sha256", "manifest_version",
        "native_trial_count", "open3d_parameter_sha256", "open3d_version",
        "pcl_parameter_sha256", "pcl_version", "planned_snapshot_count",
        "planned_trial_count", "raw_result_manifest_schema",
        "scientific_survival_commit", "scientific_survival_tag",
        "snapshot_cache_root",
        *(f"{name}_path" for name in alias_names),
        *(f"{name}_sha256" for name in alias_names),
    }
    unsigned = {
        name: item for name, item in manifest.items()
        if name != "manifest_payload_sha256"
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("manifest_payload_sha256")
        != _canonical_json_sha256(unsigned)
        or manifest.get("manifest_version") != "1"
        or manifest.get("formal_execution_authorized") is not True
        or manifest.get("formal_run_id") != "synthetic-confirmatory-v1"
        or manifest.get("formal_output_dir") != "results/synthetic_confirmatory_v1"
        or manifest.get("snapshot_cache_root")
        != "data/synthetic_confirmatory_v1_snapshots"
        or manifest.get("raw_result_manifest_schema") != RAW_MANIFEST_SCHEMA
        or manifest.get("backend_count") != 2
        or manifest.get("planned_snapshot_count") != 595
        or manifest.get("planned_trial_count") != 1190
        or manifest.get("native_trial_count") != 0
        or manifest.get("formal_workers") != 2
        or manifest.get("bootstrap_seed") != 1083684578
        or manifest.get("scientific_survival_commit")
        != "ffc15334f4ded25fdba5e709b45657dbad481dfc"
        or manifest.get("scientific_survival_tag")
        != "archive/zero-perturbation-scientific-survival-audit-v1"
        or manifest.get("open3d_parameter_sha256")
        != "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
        or manifest.get("pcl_parameter_sha256")
        != "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
        or not isinstance(manifest.get("open3d_version"), str)
        or not manifest["open3d_version"]
        or not isinstance(manifest.get("pcl_version"), str)
        or not manifest["pcl_version"]
    ):
        raise ValueError("independent formal manifest payload/identity mismatch")
    bound = manifest.get("bound_files")
    if type(bound) is not dict or set(bound) != set(_INDEPENDENT_MANIFEST_BINDINGS):
        raise ValueError("independent formal manifest binding inventory changed")
    for name, relative in _INDEPENDENT_MANIFEST_BINDINGS.items():
        row = bound.get(name)
        path = _inside(repository, relative)
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
            or not _is_sha256(row.get("sha256"))
            or not path.is_file()
            or _sha256(path) != row["sha256"]
        ):
            raise ValueError(f"independent formal manifest binding mismatch: {name}")
    for name in alias_names:
        if (
            manifest.get(f"{name}_path") != bound[name]["path"]
            or manifest.get(f"{name}_sha256") != bound[name]["sha256"]
        ):
            raise ValueError("independent formal manifest flat alias mismatch")
    return repository, manifest


def _independent_validate_snapshot_lock(
    lock_path: str | Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    expected_snapshot_count: int = 595,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the runtime lock without importing the production reader."""

    path = Path(lock_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("independent Confirmatory snapshot lock is missing")
    value = _strict_object(path)
    if set(value) != _SNAPSHOT_LOCK_FIELDS:
        raise ValueError("independent snapshot lock fields changed")
    unsigned = {
        name: item for name, item in value.items()
        if name != "snapshot_lock_payload_sha256"
    }
    entries = value.get("snapshots")
    expected_ids = [str(plan["planned_snapshot_id"]) for plan in plans]
    condition_counts = dict(sorted(Counter(
        str(plan["condition"]) for plan in plans
    ).items()))
    expected_rng_count = sum(
        3 if plan["condition"] == "FULL_NOISE" else 0 for plan in plans
    )
    if (
        len(plans) != expected_snapshot_count
        or value.get("snapshot_lock_payload_sha256")
        != _canonical_json_sha256(unsigned)
        or value.get("schema_version") != _SNAPSHOT_LOCK_SCHEMA
        or value.get("snapshot_builder_contract_version")
        != _SNAPSHOT_BUILDER_VERSION
        or type(value.get("planned_snapshot_count")) is not int
        or value["planned_snapshot_count"] != expected_snapshot_count
        or value.get("condition_snapshot_counts") != condition_counts
        or value.get("confirmatory_rng_instantiation_count")
        != expected_rng_count
        or type(entries) is not list
        or len(entries) != expected_snapshot_count
        or [entry.get("snapshot_id") if type(entry) is dict else None for entry in entries]
        != expected_ids
        or len(set(expected_ids)) != expected_snapshot_count
    ):
        raise ValueError("independent snapshot lock identity/inventory mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for plan, entry in zip(plans, entries):
        if type(entry) is not dict or set(entry) != _SNAPSHOT_LOCK_ENTRY_FIELDS:
            raise ValueError("independent snapshot lock entry fields changed")
        snapshot_id = str(plan["planned_snapshot_id"])
        exact = {
            "condition": plan["condition"],
            "confirmatory_rng_instantiation_count": (
                3 if plan["condition"] == "FULL_NOISE" else 0
            ),
            "geometry_seed": int(plan["geometry_seed"]),
            "measurement_seed": plan["measurement_seed"],
            "repeat_index": int(plan["repeat_index"]),
            "scene_variant": plan["scene_variant"],
            "snapshot_id": snapshot_id,
        }
        file_sha = entry.get("file_sha256")
        if (
            any(entry.get(name) != expected for name, expected in exact.items())
            or type(file_sha) is not dict
            or set(file_sha) != _SNAPSHOT_FILES
            or not all(_is_sha256(digest) for digest in file_sha.values())
            or not all(
                _is_sha256(entry.get(name)) for name in (
                    "metadata_payload_sha256", "reference_pose_checksum",
                    "snapshot_checksum", "source_checksum", "target_checksum",
                )
            )
        ):
            raise ValueError("independent snapshot lock entry is invalid")
        by_id[snapshot_id] = dict(entry)
    return value, by_id


def _independent_read_snapshot(
    cache_root: str | Path,
    plan: Mapping[str, Any],
    *,
    expected_lock_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate metadata, arrays, checksums, and lock binding."""

    root = Path(cache_root).resolve()
    snapshot_id = str(plan["planned_snapshot_id"])
    directory = (root / snapshot_id).resolve()
    if root not in directory.parents:
        raise ValueError("independent snapshot path escaped cache")
    if (
        not directory.is_dir()
        or {candidate.name for candidate in directory.iterdir()} != _SNAPSHOT_FILES
        or not all((directory / name).is_file() for name in _SNAPSHOT_FILES)
    ):
        raise ValueError("independent snapshot four-file inventory mismatch")
    metadata = _strict_object(directory / "metadata.json")
    if set(metadata) != _SNAPSHOT_METADATA_FIELDS:
        raise ValueError("independent snapshot metadata fields changed")
    unsigned = dict(metadata)
    stored_metadata_sha = unsigned.pop("metadata_payload_sha256")
    if stored_metadata_sha != _canonical_json_sha256(unsigned):
        raise ValueError("independent snapshot metadata payload SHA mismatch")
    condition = str(plan["condition"])
    exact = {
        "condition": condition,
        "geometry_seed": int(plan["geometry_seed"]),
        "measurement_seed": plan["measurement_seed"],
        "planned_snapshot_id": snapshot_id,
        "repeat_index": int(plan["repeat_index"]),
        "scene_variant": plan["scene_variant"],
        "snapshot_id": snapshot_id,
    }
    condition_contract = _CONDITION_METADATA.get(condition)
    if (
        condition_contract is None
        or any(metadata.get(name) != expected for name, expected in exact.items())
        or metadata.get("initial_pose") != "reference_pose_exact"
        or metadata.get("snapshot_builder_contract_version")
        != _SNAPSHOT_BUILDER_VERSION
        or metadata.get("confirmatory_rng_instantiation_count")
        != (3 if condition == "FULL_NOISE" else 0)
        or metadata.get("independent_sampling")
        is not condition_contract["independent_sampling"]
        or metadata.get("dropout_parameters")
        != condition_contract["dropout_parameters"]
        or metadata.get("noise_parameters") != condition_contract["noise_parameters"]
        or not _is_sha256(metadata.get("development_protocol_sha256"))
        or not _is_sha256(metadata.get("generator_sha256"))
        or type(metadata.get("source_point_count")) is not int
        or type(metadata.get("target_point_count")) is not int
        or metadata["source_point_count"] <= 0
        or metadata["target_point_count"] <= 0
        or type(metadata.get("source_is_target_subset")) is not bool
    ):
        raise ValueError("independent snapshot metadata/plan contract mismatch")
    file_sha = {
        name: _sha256(directory / name) for name in sorted(_SNAPSHOT_FILES)
    }
    if (
        metadata.get("array_file_sha256")
        != {
            name: file_sha[name] for name in (
                "reference_pose.npy", "source_points.npy", "target_points.npy"
            )
        }
        or type(expected_lock_entry) is not dict
        or set(expected_lock_entry) != _SNAPSHOT_LOCK_ENTRY_FIELDS
        or expected_lock_entry.get("snapshot_id") != snapshot_id
        or expected_lock_entry.get("file_sha256") != file_sha
    ):
        raise ValueError("independent snapshot file/lock SHA mismatch")
    try:
        source = np.load(directory / "source_points.npy", allow_pickle=False)
        target = np.load(directory / "target_points.npy", allow_pickle=False)
        reference = np.load(directory / "reference_pose.npy", allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("independent snapshot array load failed") from error
    if (
        source.dtype != np.dtype("<f4")
        or target.dtype != np.dtype("<f4")
        or reference.dtype != np.dtype("<f8")
        or source.ndim != 2
        or source.shape[1:] != (3,)
        or target.ndim != 2
        or target.shape[1:] != (3,)
        or reference.shape != (4, 4)
        or not all(array.flags.c_contiguous for array in (source, target, reference))
        or not all(np.all(np.isfinite(array)) for array in (source, target, reference))
        or metadata["source_point_count"] != len(source)
        or metadata["target_point_count"] != len(target)
    ):
        raise ValueError("independent snapshot array dtype/shape/finite mismatch")
    raw = {
        "source_checksum": _raw_array_sha256(source),
        "target_checksum": _raw_array_sha256(target),
        "reference_pose_checksum": _raw_array_sha256(reference),
    }
    snapshot_checksum = _canonical_json_sha256(
        {"snapshot_id": snapshot_id, **raw}
    )
    if (
        any(metadata.get(name) != digest for name, digest in raw.items())
        or metadata.get("snapshot_checksum") != snapshot_checksum
        or any(expected_lock_entry.get(name) != digest for name, digest in {
            **raw,
            "metadata_payload_sha256": stored_metadata_sha,
            "snapshot_checksum": snapshot_checksum,
        }.items())
    ):
        raise ValueError("independent snapshot raw/aggregate checksum mismatch")
    source_world = np.ascontiguousarray(
        source.astype(np.float64) @ reference[:3, :3].T + reference[:3, 3],
        dtype="<f4",
    )
    target_rows = {point.tobytes() for point in target}
    is_subset = all(point.tobytes() in target_rows for point in source_world)
    if (
        is_subset != (condition == "IDEAL_MATCHED")
        or metadata["source_is_target_subset"] != is_subset
    ):
        raise ValueError("independent snapshot sampling relationship mismatch")
    return {
        "directory": directory,
        "file_sha256": file_sha,
        "metadata": metadata,
        "source": source,
        "target": target,
        "reference": reference,
        "snapshot_checksum": snapshot_checksum,
        **raw,
    }


def _confirmatory_trial(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or value.get("condition") not in CONDITIONS:
        raise ValueError("independent verifier found unauthorized condition")
    condition = value["condition"]
    normalized = dict(value)
    normalized["condition"] = "IDEAL_MATCHED"
    checked = validate_phase_a_trial_result_strict(normalized)
    checked["condition"] = condition
    return checked


def _independent_points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty Nx3 array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return np.ascontiguousarray(array)


def _independent_transform(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0
    ):
        raise ValueError(f"{label} has an invalid homogeneous row")
    return np.ascontiguousarray(matrix)


def _independent_association(
    source: np.ndarray,
    transform: np.ndarray,
    tree: cKDTree,
    target_count: int,
) -> dict[str, np.ndarray | int]:
    transformed = np.ascontiguousarray(
        source @ transform[:3, :3].T + transform[:3, 3], dtype=np.float64
    )
    distances, targets = tree.query(transformed, k=1, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    accepted = (
        np.isfinite(distances)
        & (distances <= _ASSOCIATION_LIMIT_M)
        & (targets >= 0)
        & (targets < int(target_count))
    )
    source_indices = np.flatnonzero(accepted).astype(np.int64, copy=False)
    return {
        "source_indices": source_indices,
        "target_indices": targets[accepted],
        "source_points_target": transformed[accepted],
        "distances_m": distances[accepted],
        "count": int(source_indices.size),
    }


def _independent_target_normals(
    target: np.ndarray, tree: cKDTree
) -> tuple[np.ndarray, np.ndarray]:
    count = target.shape[0]
    normals = np.full((count, 3), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    neighbor_count = min(_PCA_NEIGHBORS, count)
    if neighbor_count < _PCA_MIN_NEIGHBORS:
        return normals, valid
    for start in range(0, count, _PCA_CHUNK):
        stop = min(count, start + _PCA_CHUNK)
        _, indices = tree.query(target[start:stop], k=neighbor_count, workers=1)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
        neighborhoods = target[indices]
        centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
        covariance = np.einsum(
            "nki,nkj->nij", centered, centered, optimize=True
        ) / float(neighbor_count)
        _, eigenvectors = np.linalg.eigh(covariance)
        candidate = eigenvectors[:, :, 0]
        norms = np.linalg.norm(candidate, axis=1)
        chunk_valid = (
            np.all(np.isfinite(candidate), axis=1)
            & np.isfinite(norms)
            & (norms > _NORMAL_EPSILON)
        )
        candidate[chunk_valid] /= norms[chunk_valid, None]
        normals[start:stop][chunk_valid] = candidate[chunk_valid]
        valid[start:stop] = chunk_valid
    return normals, valid


def _independent_residuals(
    target: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target_indices = np.asarray(state["target_indices"], dtype=np.int64)
    valid = normal_valid[target_indices]
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), valid
    targets = target[target_indices[valid]]
    chosen_normals = normals[target_indices[valid]]
    transformed = np.asarray(state["source_points_target"], dtype=np.float64)
    signed = np.einsum(
        "ij,ij->i", chosen_normals, transformed[valid] - targets
    )
    return np.asarray(signed, dtype=np.float64), valid


def _independent_residual_summary(
    signed: np.ndarray, prefix: str
) -> dict[str, float | None]:
    if signed.size == 0:
        return {
            f"{prefix}_residual_rmse": None,
            f"{prefix}_residual_median": None,
            f"{prefix}_residual_q95": None,
        }
    absolute = np.abs(signed)
    return {
        f"{prefix}_residual_rmse": float(np.sqrt(np.mean(np.square(signed)))),
        f"{prefix}_residual_median": float(np.median(absolute)),
        f"{prefix}_residual_q95": float(
            np.quantile(absolute, 0.95, method="linear")
        ),
    }


def _independent_translation_geometry(
    normals: np.ndarray, signed: np.ndarray
) -> dict[str, float | None]:
    if signed.size == 0:
        return {
            "lambda_min_trans": None,
            "lambda_mid_trans": None,
            "lambda_max_trans": None,
            "normalized_lambda_min_trans": None,
            "normalized_lambda_mid_trans": None,
            "normalized_lambda_max_trans": None,
            "condition_number_trans": None,
            "spectral_entropy_trans": None,
            "initial_translation_gradient_norm": None,
        }
    hessian = (normals.T @ normals) / float(normals.shape[0])
    eigenvalues = np.maximum(np.linalg.eigvalsh(hessian), 0.0)
    total = max(float(np.sum(eigenvalues)), 1.0e-12)
    normalized = eigenvalues / total
    positive = normalized > 0.0
    entropy = -float(
        np.sum(normalized[positive] * np.log(normalized[positive]))
    ) / math.log(3.0)
    gradient = np.mean(normals * signed[:, None], axis=0)
    return {
        "lambda_min_trans": float(eigenvalues[0]),
        "lambda_mid_trans": float(eigenvalues[1]),
        "lambda_max_trans": float(eigenvalues[2]),
        "normalized_lambda_min_trans": float(normalized[0]),
        "normalized_lambda_mid_trans": float(normalized[1]),
        "normalized_lambda_max_trans": float(normalized[2]),
        "condition_number_trans": float(
            eigenvalues[2] / max(float(eigenvalues[0]), 1.0e-12)
        ),
        "spectral_entropy_trans": entropy,
        "initial_translation_gradient_norm": float(np.linalg.norm(gradient)),
    }


def _independent_prepare_common(
    source_value: np.ndarray,
    target_value: np.ndarray,
    reference_value: np.ndarray,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    source = np.array(
        _independent_points(source_value, "source_points"),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    target = np.array(
        _independent_points(target_value, "target_points"),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    reference = np.array(
        _independent_transform(reference_value, "reference_transform"),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    tree = cKDTree(target)
    normals, normal_valid = _independent_target_normals(target, tree)
    initial = _independent_association(
        source, reference, tree, target.shape[0]
    )
    signed, valid_matches = _independent_residuals(
        target, normals, normal_valid, initial
    )
    targets = np.asarray(initial["target_indices"], dtype=np.int64)
    selected_normals = normals[targets[valid_matches]]
    initial_metrics = {
        "initial_correspondence_count": int(initial["count"]),
        "initial_valid_normal_correspondence_count": int(signed.size),
        "target_normal_valid_count": int(np.count_nonzero(normal_valid)),
        "target_normal_invalid_count": int(
            normal_valid.size - np.count_nonzero(normal_valid)
        ),
        **_independent_residual_summary(signed, "initial"),
        **_independent_translation_geometry(selected_normals, signed),
    }
    return {
        "source": source,
        "target": target,
        "reference": reference,
        "tree": tree,
        "normals": normals,
        "normal_valid": normal_valid,
        "initial": initial,
        "initial_metrics": initial_metrics,
        "snapshot_id": snapshot_id,
    }


def _independent_turnover(
    initial: Mapping[str, Any], final: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    initial_sources = np.asarray(initial["source_indices"], dtype=np.int64)
    final_sources = np.asarray(final["source_indices"], dtype=np.int64)
    common, initial_positions, final_positions = np.intersect1d(
        initial_sources,
        final_sources,
        assume_unique=True,
        return_indices=True,
    )
    initial_targets = np.asarray(initial["target_indices"], dtype=np.int64)
    final_targets = np.asarray(final["target_indices"], dtype=np.int64)
    same_pair_count = int(np.count_nonzero(
        initial_targets[initial_positions] == final_targets[final_positions]
    ))
    initial_count, final_count = int(initial["count"]), int(final["count"])
    pair_union = initial_count + final_count - same_pair_count
    source_union = initial_count + final_count - int(common.size)
    return (
        None if pair_union == 0 else 1.0 - same_pair_count / pair_union,
        None if source_union == 0 else 1.0 - int(common.size) / source_union,
    )


def _independent_normal_angles(
    context: Mapping[str, Any], final: Mapping[str, Any]
) -> tuple[float | None, float | None, int]:
    initial = context["initial"]
    _, initial_positions, final_positions = np.intersect1d(
        np.asarray(initial["source_indices"], dtype=np.int64),
        np.asarray(final["source_indices"], dtype=np.int64),
        assume_unique=True,
        return_indices=True,
    )
    initial_targets = np.asarray(initial["target_indices"], dtype=np.int64)[
        initial_positions
    ]
    final_targets = np.asarray(final["target_indices"], dtype=np.int64)[
        final_positions
    ]
    normal_valid = np.asarray(context["normal_valid"], dtype=bool)
    valid = normal_valid[initial_targets] & normal_valid[final_targets]
    if not np.any(valid):
        return None, None, 0
    normals = np.asarray(context["normals"], dtype=np.float64)
    cosine = np.clip(
        np.abs(np.einsum(
            "ij,ij->i",
            normals[initial_targets[valid]],
            normals[final_targets[valid]],
        )),
        0.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosine))
    return (
        float(np.median(angles)),
        float(np.quantile(angles, 0.95, method="linear")),
        int(angles.size),
    )


def _independent_invalid_reason(metrics: Mapping[str, Any]) -> str | None:
    if metrics["initial_correspondence_count"] == 0:
        return "NO_INITIAL_CORRESPONDENCE"
    if metrics["final_correspondence_count"] == 0:
        return "NO_FINAL_CORRESPONDENCE"
    if (
        metrics["initial_valid_normal_correspondence_count"] == 0
        or metrics["final_valid_normal_correspondence_count"] == 0
        or metrics["common_valid_normal_source_count"] == 0
    ):
        return "INSUFFICIENT_VALID_NORMALS"
    for name in _COMMON_METRIC_FIELDS:
        value = metrics.get(name)
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
            return "NONFINITE_COMMON_METRICS"
    return None


def _independent_common_record(
    context: Mapping[str, Any],
    estimated_value: np.ndarray,
    *,
    identifiers: Mapping[str, Any],
) -> dict[str, Any]:
    estimated = _independent_transform(estimated_value, "estimated_transform")
    final = _independent_association(
        context["source"], estimated, context["tree"], context["target"].shape[0]
    )
    final_signed, _ = _independent_residuals(
        context["target"], context["normals"], context["normal_valid"], final
    )
    final_residual = _independent_residual_summary(final_signed, "final")
    pair_turnover, source_turnover = _independent_turnover(
        context["initial"], final
    )
    median_angle, q95_angle, common_normal_count = _independent_normal_angles(
        context, final
    )
    initial_count = int(context["initial"]["count"])
    initial_rmse = context["initial_metrics"]["initial_residual_rmse"]
    final_rmse = final_residual["final_residual_rmse"]
    metrics = {
        **dict(context["initial_metrics"]),
        "final_correspondence_count": int(final["count"]),
        "final_valid_normal_correspondence_count": int(final_signed.size),
        "common_valid_normal_source_count": common_normal_count,
        "correspondence_turnover": pair_turnover,
        "accepted_source_turnover": source_turnover,
        "correspondence_count_change_ratio": (
            None if initial_count == 0
            else float((int(final["count"]) - initial_count) / initial_count)
        ),
        **final_residual,
        "residual_rmse_change": (
            None if initial_rmse is None or final_rmse is None
            else float(final_rmse - initial_rmse)
        ),
        "median_normal_angle_change_deg": median_angle,
        "q95_normal_angle_change_deg": q95_angle,
    }
    reason = _independent_invalid_reason(metrics)
    return {
        **dict(identifiers),
        "snapshot_id": context["snapshot_id"],
        "common_association_valid": reason is None,
        "common_association_invalid_reason": reason,
        "common_association_is_backend_internal": False,
        "association_distance_limit_m": _ASSOCIATION_LIMIT_M,
        "target_normal_pca_k": _PCA_NEIGHBORS,
        "target_normal_pca_min_neighbors": _PCA_MIN_NEIGHBORS,
        **metrics,
    }


def _independent_safe_common_record(
    context: Mapping[str, Any],
    estimated: np.ndarray,
    *,
    identifiers: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _independent_common_record(
            context, estimated, identifiers=identifiers
        )
    except Exception as error:
        normal_valid = np.asarray(context["normal_valid"], dtype=bool)
        return {
            **dict(identifiers),
            "snapshot_id": context["snapshot_id"],
            "common_association_valid": False,
            "common_association_invalid_reason": "OTHER",
            "common_association_invalid_detail": f"{type(error).__name__}: {error}",
            "common_association_is_backend_internal": False,
            "association_distance_limit_m": _ASSOCIATION_LIMIT_M,
            "target_normal_pca_k": _PCA_NEIGHBORS,
            "target_normal_pca_min_neighbors": _PCA_MIN_NEIGHBORS,
            "target_normal_valid_count": int(np.count_nonzero(normal_valid)),
            "target_normal_invalid_count": int(
                normal_valid.size - np.count_nonzero(normal_valid)
            ),
            "common_valid_normal_source_count": 0,
            **{name: None for name in _COMMON_METRIC_FIELDS},
        }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return result


def _rho(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x = np.asarray(rankdata(left, method="average"), dtype=np.float64)
    y = np.asarray(rankdata(right, method="average"), dtype=np.float64)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    result = float(np.corrcoef(x, y)[0, 1])
    return result if math.isfinite(result) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _q95(values: Sequence[float]) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), .95, method="linear")) if values else None


def _feature(row: Mapping[str, Any], name: str) -> float:
    if name == "log10_initial_residual_rmse_plus_1e-9":
        value = _number(row.get("initial_residual_rmse"))
        return math.log10(value + 1e-9) if value is not None and value >= 0 else math.nan
    if name == "log10_condition_number_trans_plus_1":
        value = _number(row.get("condition_number_trans"))
        return math.log10(value + 1.0) if value is not None and value >= 0 else math.nan
    if name == "log10_inverse_lambda_min_trans":
        value = _number(row.get("lambda_min_trans"))
        return math.log10(1.0 / max(value, 1e-12)) if value is not None else math.nan
    if name == "log10_initial_correspondence_count_plus_1":
        value = _number(row.get("initial_correspondence_count"))
        return math.log10(value + 1.0) if value is not None and value >= 0 else math.nan
    value = _number(row.get(name))
    return value if value is not None else math.nan


def _locked_prediction(model: Mapping[str, Any], row: Mapping[str, Any]) -> float | None:
    names = model.get("feature_names")
    if type(names) is not list:
        return None
    vector = np.asarray([_feature(row, str(name)) for name in names], dtype=np.float64)
    mean = np.asarray(model.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(model.get("scaler_scale"), dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient"), dtype=np.float64)
    intercept = _number(model.get("intercept"))
    if any(item.shape != vector.shape for item in (mean, scale, coefficient)) or intercept is None or not all(np.all(np.isfinite(item)) for item in (vector, mean, scale, coefficient)) or np.any(scale <= 0):
        return None
    return float(np.dot((vector - mean) / scale, coefficient) + intercept)


def _rows(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    identifiers = []
    for source in trials:
        row = dict(source)
        backend = str(row.get("backend_schema_name", row.get("backend", "")))
        condition, scene = str(row.get("condition", "")), str(row.get("scene_variant", ""))
        if backend not in BACKENDS or condition not in CONDITIONS or scene not in SCENES:
            raise ValueError("independent verifier found unauthorized identity")
        trial_id = str(row.get("planned_trial_id", ""))
        snapshot_id = str(row.get("planned_snapshot_id", row.get("snapshot_id", "")))
        if not trial_id or not snapshot_id:
            raise ValueError("independent verifier found missing identity")
        vector = row.get("translation_vector")
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("independent translation vector is invalid")
            vector = vector.astype(float).tolist()
        output.append({
            **row, "backend_schema_name": backend, "condition": condition,
            "scene_variant": scene, "planned_trial_id": trial_id,
            "planned_snapshot_id": snapshot_id, "geometry_seed": int(row["geometry_seed"]),
            "measurement_seed": None if row.get("measurement_seed") in (None, "") else int(row["measurement_seed"]),
            "repeat_index": int(row.get("repeat_index", 0)),
            "solver_failure": bool(row.get("solver_failure", False)),
            "finite_output": bool(row.get("finite_output", True)),
            "translation_error_m": _number(row.get("translation_error_m", row.get("translation_update_m"))),
            "rotation_error_rad": _number(row.get("rotation_error_rad", row.get("rotation_update_rad"))),
            "translation_vector": vector,
        })
        identifiers.append(trial_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("independent verifier found duplicate trials")
    return sorted(output, key=lambda row: row["planned_trial_id"])


def independently_recompute_synthetic_confirmatory(
    *, trials: Sequence[Mapping[str, Any]], common_records: Sequence[Mapping[str, Any]],
    model_lock: Mapping[str, Any], gate_contract: Mapping[str, Any],
    expected_geometry_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    rows = _rows(trials)
    geometries = tuple(int(value) for value in (expected_geometry_seeds if expected_geometry_seeds is not None else sorted({row["geometry_seed"] for row in rows})))
    if len(geometries) != 5 or len(set(geometries)) != 5:
        raise ValueError("independent verifier requires five geometry seeds")
    gates = gate_contract.get("hypotheses")
    if type(gates) is not dict or len(gates) != 6:
        raise ValueError("independent H1--H6 contract is invalid")
    counts = Counter((r["backend_schema_name"], r["scene_variant"], r["geometry_seed"], r["condition"]) for r in rows)
    exact = len(rows) == 1190 and len(counts) == 2 * 7 * 5 * 3 and all(value == {"IDEAL_MATCHED": 1, "INDEPENDENT_NOISE_FREE": 1, "FULL_NOISE": 15}[key[3]] for key, value in counts.items())
    usable = lambda row: not row["solver_failure"] and row["finite_output"] and row["translation_error_m"] is not None and row["rotation_error_rad"] is not None

    h1 = []
    c1 = gates["H1_IDEAL_CONTROL"]
    for backend in BACKENDS:
        chosen = [r for r in rows if r["backend_schema_name"] == backend and r["condition"] == "IDEAL_MATCHED"]
        t = [float(r["translation_error_m"]) for r in chosen if usable(r)]
        q = [float(r["rotation_error_rad"]) for r in chosen if usable(r)]
        tq, rq = _q95(t), _q95(q)
        sf, nf = sum(r["solver_failure"] for r in chosen), sum(not r["finite_output"] for r in chosen)
        passed = len(chosen) == len(t) == 35 and sf <= c1["solver_failure_count_max"] and nf <= c1["nonfinite_output_count_max"] and tq is not None and tq <= c1["translation_q95_max_m"] and rq is not None and rq <= c1["rotation_q95_max_rad"]
        h1.append({"backend_schema_name": backend, "planned_count": len(chosen), "successful_count": len(t), "solver_failure_count": sf, "nonfinite_output_count": nf, "translation_q95_m": tq, "rotation_q95_rad": rq, "gate_pass": bool(passed)})

    h2 = []
    c2 = gates["H2_LONG_CORRIDOR_SCENE_EFFECT"]
    for backend in BACKENDS:
        for condition in c2["conditions"]:
            detail = []
            for geometry in geometries:
                values = {}
                for scene in ("GEOMETRY_RICH_ROOM", "LONG_CORRIDOR"):
                    values[scene] = [float(r["translation_error_m"]) for r in rows if r["backend_schema_name"] == backend and r["condition"] == condition and r["geometry_seed"] == geometry and r["scene_variant"] == scene and usable(r)]
                rich, corridor = _median(values["GEOMETRY_RICH_ROOM"]), _median(values["LONG_CORRIDOR"])
                valid = rich is not None and corridor is not None
                ratio = corridor / rich if valid and rich > 0 else (np.finfo(np.float64).max if valid and corridor > 0 else None)
                difference = corridor - rich if valid else None
                detail.append({"geometry_seed": geometry, "rich_median_m": rich, "corridor_median_m": corridor, "ratio": ratio, "difference_m": difference, "corridor_greater": bool(valid and corridor > rich)})
            ratios = [float(r["ratio"]) for r in detail if r["ratio"] is not None]
            diffs = [float(r["difference_m"]) for r in detail if r["difference_m"] is not None]
            wins, mr, md = sum(r["corridor_greater"] for r in detail), _median(ratios), _median(diffs)
            passed = len(ratios) == c2["geometry_block_count"] and wins >= c2["minimum_long_greater_than_rich_blocks"] and mr is not None and mr >= c2["geometry_level_median_ratio_min"] and md is not None and md >= c2["geometry_level_median_absolute_difference_min_m"]
            h2.append({"backend_schema_name": backend, "condition": condition, "geometry_rows": detail, "passing_geometry_count": wins, "median_geometry_ratio": mr, "median_geometry_difference_m": md, "gate_pass": bool(passed)})

    h3 = []
    c3 = gates["H3_CROSS_BACKEND_SCENE_RANKING"]
    for condition in c3["conditions"]:
        medians = {backend: [] for backend in BACKENDS}
        for backend in BACKENDS:
            for scene in SCENES:
                medians[backend].append(_median([float(r["translation_error_m"]) for r in rows if r["backend_schema_name"] == backend and r["condition"] == condition and r["scene_variant"] == scene and usable(r)]))
        valid = all(value is not None for values in medians.values() for value in values)
        rho = _rho(medians[BACKENDS[0]], medians[BACKENDS[1]]) if valid else None
        h3.append({"condition": condition, "spearman_rho": rho, "scene_medians": medians, "gate_pass": bool(rho is not None and rho >= c3["spearman_rho_min"] and len(medians[BACKENDS[0]]) == c3["scene_count"])})

    common = [dict(r) for r in common_records]
    common_ids = [str(r.get("planned_trial_id", "")) for r in common]
    if any(not value for value in common_ids) or len(common_ids) != len(set(common_ids)):
        raise ValueError("independent common inventory is invalid")
    by_id = {str(r["planned_trial_id"]): r for r in common}
    eligible = {r["planned_trial_id"] for r in rows if r["condition"] != "IDEAL_MATCHED" and usable(r)}
    common_exact = set(by_id) == eligible
    joined = [{**r, **by_id[r["planned_trial_id"]], "backend_schema_name": r["backend_schema_name"]} for r in rows if r["planned_trial_id"] in by_id]

    h4 = []
    c4 = gates["H4_REASSOCIATION_MECHANISM"]
    for backend in BACKENDS:
        chosen = [r for r in joined if r["backend_schema_name"] == backend and r.get("common_association_valid") is True and _number(r.get("correspondence_turnover")) is not None]
        pooled = _rho(
            [float(r["correspondence_turnover"]) for r in chosen],
            [math.log10(float(r["translation_error_m"]) + EPSILON) for r in chosen],
        )
        cx, cy = [], []
        grouped = defaultdict(list)
        for r in chosen: grouped[(r["scene_variant"], r["condition"])].append(r)
        for group in grouped.values():
            mx = np.median([float(r["correspondence_turnover"]) for r in group])
            my = np.median([
                math.log10(float(r["translation_error_m"]) + EPSILON)
                for r in group
            ])
            cx += [float(r["correspondence_turnover"]) - mx for r in group]
            cy += [
                math.log10(float(r["translation_error_m"]) + EPSILON) - my
                for r in group
            ]
        centered = _rho(cx, cy)
        folds = []
        for held in geometries:
            training = [r for r in chosen if r["geometry_seed"] != held]
            value = _rho(
                [float(r["correspondence_turnover"]) for r in training],
                [
                    math.log10(float(r["translation_error_m"]) + EPSILON)
                    for r in training
                ],
            )
            folds.append({"held_out_geometry_seed": held, "spearman_rho": value, "direction_positive": bool(value is not None and value > 0)})
        passed = pooled is not None and pooled >= c4["pooled_turnover_error_spearman_rho_min"] and centered is not None and centered >= c4["scene_centered_spearman_rho_min"] and len(folds) == c4["leave_one_geometry_seed_out_fold_count"] and all(r["direction_positive"] for r in folds)
        h4.append({
            "backend_schema_name": backend,
            "valid_trial_count": len(chosen),
            "error_transform": "log10(translation_error_m+1e-9)",
            "centering_cell": ["scene_variant", "condition"],
            "pooled_spearman_rho": pooled,
            "scene_centered_spearman_rho": centered,
            "leave_one_geometry_out": folds,
            "gate_pass": bool(passed),
        })

    models = model_lock.get("models")
    if type(models) is not list:
        raise ValueError("independent model lock is invalid")
    model_map = {(str(m.get("backend_schema_name")), str(m.get("model"))): m for m in models}
    if set(model_map) != {(backend, model) for backend in BACKENDS for model in ("A", "B")}:
        raise ValueError("independent model identities changed")
    h5 = []
    c5 = gates["H5_FROZEN_MODEL_B_INCREMENTAL_VALUE"]
    for backend in BACKENDS:
        target, pa, pb = [], [], []
        selected = [
            r for r in rows
            if r["backend_schema_name"] == backend
            and r["condition"] != "IDEAL_MATCHED"
            and usable(r)
        ]
        candidates = len(selected)
        invalid = 0
        for trial in selected:
            metric = by_id.get(trial["planned_trial_id"])
            if metric is None:
                invalid += 1
                continue
            r = {
                **trial,
                **metric,
                "backend_schema_name": trial["backend_schema_name"],
            }
            if r.get("common_association_valid") is not True:
                invalid += 1
                continue
            error = _number(r.get("translation_error_m")); a = _locked_prediction(model_map[(backend, "A")], r); b = _locked_prediction(model_map[(backend, "B")], r)
            if error is not None and error >= 0 and a is not None and b is not None:
                target.append(math.log10(error + EPSILON)); pa.append(a); pb.append(b)
            else:
                invalid += 1
        ma = float(np.mean(np.abs(np.asarray(pa) - np.asarray(target)))) if target else None
        mb = float(np.mean(np.abs(np.asarray(pb) - np.asarray(target)))) if target else None
        ratio = mb / ma if ma is not None and ma > 0 and mb is not None else None
        h5.append({
            "backend_schema_name": backend,
            "candidate_count": candidates,
            "evaluated_count": len(target),
            "invalid_count": invalid,
            "model_a_mae": ma,
            "model_b_mae": mb,
            "mae_b_to_a_ratio": ratio,
            "gate_pass": bool(
                candidates > 0 and invalid == 0 and len(target) == candidates
                and ratio is not None
                and ratio <= c5["mae_b_to_mae_a_ratio_max"]
            ),
        })

    h6groups = []
    c6 = gates["H6_FULL_NOISE_SYSTEMATIC_OFFSET"]
    for backend in BACKENDS:
        for geometry in geometries:
            chosen = [r for r in rows if r["backend_schema_name"] == backend and r["condition"] == c6["condition"] and r["scene_variant"] == c6["scene_variant"] and r["geometry_seed"] == geometry and usable(r) and r.get("translation_vector") is not None]
            unique = {(str(r.get("source_checksum", r["planned_snapshot_id"])), str(r.get("target_checksum", ""))) for r in chosen}
            vectors = np.asarray([r["translation_vector"] for r in chosen], dtype=np.float64)
            offset = float(np.linalg.norm(np.mean(vectors, axis=0))) if len(vectors) else None
            mean_norm = float(np.mean(np.linalg.norm(vectors, axis=1))) if len(vectors) else None
            fraction = offset / mean_norm if offset is not None and mean_norm is not None and mean_norm > 0 else None
            passed = len(unique) >= c6["effective_replicate_count_min"] and offset is not None and offset >= c6["systematic_translation_offset_min_m"] and fraction is not None and fraction >= c6["systematic_fraction_min"]
            h6groups.append({"backend_schema_name": backend, "geometry_seed": geometry, "successful_count": len(chosen), "effective_replicate_count": len(unique), "systematic_translation_offset_m": offset, "systematic_fraction": fraction, "group_gate_pass": bool(passed)})
    h6backend = []
    for backend in BACKENDS:
        chosen = [r for r in h6groups if r["backend_schema_name"] == backend]
        fractions = [float(r["systematic_fraction"]) for r in chosen if r["systematic_fraction"] is not None]
        passing, median_fraction = sum(r["group_gate_pass"] for r in chosen), _median(fractions)
        h6backend.append({"backend_schema_name": backend, "passing_geometry_group_count": passing, "median_systematic_fraction": median_fraction, "gate_pass": bool(len(chosen) == c6["geometry_group_count"] and passing >= c6["minimum_passing_geometry_groups"] and median_fraction is not None and median_fraction >= c6["median_systematic_fraction_min"])})

    summary = {
        "CONFIRMATORY_EVIDENCE_INTEGRITY_PASS": bool(exact and common_exact),
        "H1_IDEAL_CONTROL_PASS": all(r["gate_pass"] for r in h1),
        "H2_LONG_CORRIDOR_SCENE_EFFECT_PASS": len(h2) == 4 and all(r["gate_pass"] for r in h2),
        "H3_CROSS_BACKEND_SCENE_RANKING_PASS": len(h3) == 2 and all(r["gate_pass"] for r in h3),
        "H4_REASSOCIATION_MECHANISM_PASS": common_exact and all(r["gate_pass"] for r in h4),
        "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS": common_exact and all(r["gate_pass"] for r in h5),
        "H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS": all(r["gate_pass"] for r in h6backend),
    }
    decision = {
        **summary,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": True,
        "SYNTHETIC_CONFIRMATORY_COMPLETE": exact,
        "SYNTHETIC_CONFIRMATORY_PASS": all(summary.values()),
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }
    projection = {
        "final_decision": decision, "gate_summary": summary,
        "integrity": {"planned_trial_count": 1190, "observed_trial_count": len(rows), "exact_plan_cardinality_pass": exact, "common_expected_count": len(eligible), "common_observed_count": len(common), "common_record_inventory_pass": common_exact},
        "h1_ideal_control": h1, "h2_scene_effect": h2,
        "h3_cross_backend_ranking": h3, "h4_reassociation": h4,
        "h5_frozen_models": h5, "h6_systematic_groups": h6groups,
        "h6_systematic_backend": h6backend,
    }
    return {"schema_version": "synthetic_confirmatory_independent_verification_v1", "verification_projection": projection, "final_decision": decision}


def _plan_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError("independent plan CSV schema changed")
            raw = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"independent plan CSV is invalid: {path}") from error
    output = []
    for row in raw:
        if None in row:
            raise ValueError("independent plan CSV has excess columns")
        output.append({
            **row,
            "geometry_seed": int(row["geometry_seed"]),
            "measurement_seed": (
                None if row["measurement_seed"] == ""
                else int(row["measurement_seed"])
            ),
            "repeat_index": int(row["repeat_index"]),
        })
    return output


def independently_read_synthetic_confirmatory_evidence(
    *, manifest_path: str | Path, run_dir: str | Path
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    """Independently re-read raw JSON, snapshot metadata/arrays, and metrics."""

    from .rotation_metrics import rotation_metric_audit

    repository, manifest = _independent_validate_formal_manifest(manifest_path)

    trial_fields = (
        "planned_trial_id", "planned_snapshot_id", "scene_variant", "condition",
        "geometry_seed", "measurement_seed", "repeat_index", "backend",
    )
    snapshot_fields = (
        "planned_snapshot_id", "scene_variant", "condition", "geometry_seed",
        "measurement_seed", "repeat_index", "planned_backend_count",
        "replicate_semantics",
    )
    trial_plan_path = _inside(repository, str(manifest["planned_trials_path"]))
    snapshot_plan_path = _inside(
        repository, str(manifest["planned_snapshots_path"])
    )
    if (
        _sha256(trial_plan_path) != manifest["planned_trials_sha256"]
        or _sha256(snapshot_plan_path) != manifest["planned_snapshots_sha256"]
    ):
        raise ValueError("independent plan SHA mismatch")
    trial_plans = _plan_csv(trial_plan_path, trial_fields)
    snapshot_plans = _plan_csv(snapshot_plan_path, snapshot_fields)
    if (
        len(trial_plans) != 1190
        or len(snapshot_plans) != 595
        or len({row["planned_trial_id"] for row in trial_plans}) != 1190
        or len({row["planned_snapshot_id"] for row in snapshot_plans}) != 595
    ):
        raise ValueError("independent frozen plan inventory changed")

    directory = Path(run_dir).resolve()
    if directory != _inside(repository, str(manifest["formal_output_dir"])):
        raise ValueError("independent result directory differs from manifest")
    raw_manifest_path = directory / "raw_result_manifest.json"
    if not raw_manifest_path.is_file():
        raise FileNotFoundError("Synthetic Confirmatory raw results do not exist")
    raw_manifest = _strict_object(raw_manifest_path)
    if (
        set(raw_manifest) != {"schema_version", "run_id", "results"}
        or raw_manifest.get("schema_version") != RAW_MANIFEST_SCHEMA
        or raw_manifest.get("run_id") != manifest["formal_run_id"]
        or type(raw_manifest.get("results")) is not dict
    ):
        raise ValueError("independent raw result manifest contract mismatch")
    plans_by_id = {str(row["planned_trial_id"]): row for row in trial_plans}
    entries = raw_manifest["results"]
    if set(entries) != set(plans_by_id):
        raise ValueError("independent raw result inventory is incomplete or has extras")
    raw_root = (directory / "raw_results").resolve()
    trials_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial_id in sorted(plans_by_id):
        entry = entries[trial_id]
        if (
            type(entry) is not dict
            or set(entry) != {"planned_trial_id", "path", "sha256"}
            or entry.get("planned_trial_id") != trial_id
            or not isinstance(entry.get("path"), str)
            or Path(entry["path"]).name != entry["path"]
        ):
            raise ValueError("independent raw manifest entry contract mismatch")
        result_path = (raw_root / entry["path"]).resolve()
        if (
            result_path.parent != raw_root
            or not result_path.is_file()
            or _sha256(result_path) != entry.get("sha256")
        ):
            raise ValueError("independent raw result file/SHA mismatch")
        result = _confirmatory_trial(_strict_object(result_path))
        plan = plans_by_id[trial_id]
        exact = {
            "planned_trial_id": trial_id,
            "snapshot_id": plan["planned_snapshot_id"],
            "scene_variant": plan["scene_variant"],
            "condition": plan["condition"],
            "backend": plan["backend"],
        }
        if any(result.get(name) != expected for name, expected in exact.items()):
            raise ValueError("independent raw result differs from frozen plan")
        trials_by_snapshot[plan["planned_snapshot_id"]].append({
            **result,
            "backend_schema_name": result["backend"],
            "geometry_seed": plan["geometry_seed"],
            "measurement_seed": plan["measurement_seed"],
            "planned_snapshot_id": plan["planned_snapshot_id"],
            "repeat_index": plan["repeat_index"],
        })

    cache = _inside(repository, str(manifest["snapshot_cache_root"]))
    snapshot_lock_path = repository / "data/synthetic_confirmatory_v1_snapshot_lock.json"
    _snapshot_lock, lock_by_id = _independent_validate_snapshot_lock(
        snapshot_lock_path, snapshot_plans
    )
    snapshot_lock_sha256 = _sha256(snapshot_lock_path)
    normalized: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    snapshot_plan_by_id = {
        str(row["planned_snapshot_id"]): row for row in snapshot_plans
    }
    if set(trials_by_snapshot) != set(snapshot_plan_by_id):
        raise ValueError("independent trial/snapshot plan pairing mismatch")
    for snapshot_id in sorted(snapshot_plan_by_id):
        plan = snapshot_plan_by_id[snapshot_id]
        snapshot = _independent_read_snapshot(
            cache,
            plan,
            expected_lock_entry=lock_by_id[snapshot_id],
        )
        chosen = trials_by_snapshot[snapshot_id]
        if len(chosen) != 2 or {row["backend"] for row in chosen} != set(BACKENDS):
            raise ValueError("independent backend pairing mismatch")
        context = None
        if plan["condition"] != "IDEAL_MATCHED":
            context = _independent_prepare_common(
                snapshot["source"], snapshot["target"], snapshot["reference"],
                snapshot_id=snapshot_id,
            )
        for row in chosen:
            if any(
                row.get(name) != snapshot[name]
                for name in (
                    "snapshot_checksum", "source_checksum", "target_checksum",
                    "reference_pose_checksum",
                )
            ):
                raise ValueError("independent trial/snapshot checksum mismatch")
            if row.get("snapshot_lock_sha256") != snapshot_lock_sha256:
                raise ValueError("independent trial/snapshot-lock SHA mismatch")
            updated = dict(row)
            estimate_value = row.get("final_transform_4x4")
            if (
                estimate_value is not None
                and not row["solver_failure"]
                and row["finite_output"]
            ):
                estimate = np.asarray(estimate_value, dtype=np.float64)
                vector = estimate[:3, 3] - snapshot["reference"][:3, 3]
                translation = float(np.linalg.norm(vector))
                rotation = rotation_metric_audit(
                    estimate[:3, :3], snapshot["reference"][:3, :3]
                )["rotation_error_rad"]
                if rotation is None:
                    raise ValueError("independent rotation recomputation failed")
                if (
                    abs(float(row["translation_update_m"]) - translation) > 1e-12
                    or abs(float(row["rotation_update_rad"]) - rotation) > 1e-12
                ):
                    raise ValueError("independent stored metric mismatch")
                updated.update(
                    translation_error_m=translation,
                    translation_vector=vector.astype(float).tolist(),
                    rotation_error_rad=float(rotation),
                )
            normalized.append(updated)
            if (
                plan["condition"] != "IDEAL_MATCHED"
                and not row["solver_failure"]
                and row["finite_output"]
            ):
                if context is None or estimate_value is None:
                    raise ValueError("independent successful trial lacks common input")
                common.append(_independent_safe_common_record(
                    context,
                    np.asarray(estimate_value, dtype=np.float64),
                    identifiers={
                        "planned_trial_id": row["planned_trial_id"],
                        "backend_schema_name": row["backend_schema_name"],
                        "scene_variant": row["scene_variant"],
                        "condition": row["condition"],
                        "geometry_seed": row["geometry_seed"],
                        "measurement_seed": row["measurement_seed"],
                        "repeat_index": row["repeat_index"],
                    },
                ))
    return (
        sorted(normalized, key=lambda row: row["planned_trial_id"]),
        sorted(common, key=lambda row: row["planned_trial_id"]),
        manifest,
        raw_manifest,
    )


def independently_verify_synthetic_confirmatory(
    *, manifest_path: str | Path, run_dir: str | Path
) -> dict[str, Any]:
    trials, common, manifest, _raw_manifest = (
        independently_read_synthetic_confirmatory_evidence(
            manifest_path=manifest_path, run_dir=run_dir
        )
    )
    repository = Path(manifest_path).resolve().parent.parent
    protocol = _strict_object(
        _inside(repository, str(manifest["scientific_protocol_path"]))
    )
    result = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_strict_object(
            _inside(repository, str(manifest["frozen_model_path"]))
        ),
        gate_contract=_strict_object(
            _inside(repository, str(manifest["gate_contract_path"]))
        ),
        expected_geometry_seeds=protocol["geometry_seeds"],
    )
    result["run_id"] = manifest["formal_run_id"]
    result["raw_result_manifest_sha256"] = _sha256(
        Path(run_dir).resolve() / "raw_result_manifest.json"
    )
    return result


def primary_verification_projection(primary: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "final_decision", "gate_summary", "integrity", "h1_ideal_control",
        "h2_scene_effect", "h3_cross_backend_ranking", "h4_reassociation",
        "h5_frozen_models", "h6_systematic_groups", "h6_systematic_backend",
    )
    return {name: primary[name] for name in names}


def _compare(left: Any, right: Any) -> tuple[int, float]:
    if isinstance(left, bool) or isinstance(right, bool):
        return ((0, 0.0) if type(left) is type(right) and left is right else (1, 0.0))
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, float) or not isinstance(right, float): return 1, 0.0
        difference = abs(left - right)
        return (0 if difference == 0.0 else 1), difference
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right): return 1, 0.0
        results = [_compare(left[key], right[key]) for key in sorted(left)]
        return sum(item[0] for item in results), max((item[1] for item in results), default=0.0)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right): return 1, 0.0
        results = [_compare(a, b) for a, b in zip(left, right)]
        return sum(item[0] for item in results), max((item[1] for item in results), default=0.0)
    return ((0, 0.0) if type(left) is type(right) and left == right else (1, 0.0))


def compare_primary_and_independent(primary: Mapping[str, Any], independent: Mapping[str, Any]) -> dict[str, Any]:
    expected = primary_verification_projection(primary)
    actual = independent.get("verification_projection")
    if type(actual) is not dict:
        raise ValueError("independent verification projection is missing")
    sections, leaves, maximum, names = 0, 0, 0.0, []
    for name in expected:
        count, difference = _compare(expected[name], actual.get(name))
        if count: sections += 1; names.append(name)
        leaves += count; maximum = max(maximum, difference)
    return {
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "section_difference_count": sections,
        "leaf_difference_count": leaves,
        "maximum_absolute_numeric_difference": maximum,
        "exact_match_pass": leaves == 0 and maximum == 0.0,
        "differing_sections": names,
    }


__all__ = [
    "compare_primary_and_independent",
    "independently_read_synthetic_confirmatory_evidence",
    "independently_recompute_synthetic_confirmatory",
    "independently_verify_synthetic_confirmatory",
    "primary_verification_projection",
]
