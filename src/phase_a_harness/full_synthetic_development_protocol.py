"""One frozen protocol and plan for Full Synthetic Development v1.

The module owns only experiment identity, read-only Phase A import validation,
plan construction, and the single manifest's one-way authorization transition.
It never generates a point cloud or executes a registration backend.
"""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    OPEN3D_PLAN_BACKEND,
    PCL_PLAN_BACKEND,
    canonical_json_sha256,
    file_sha256,
    load_manifest,
    manifest_root,
    write_json,
)
from .phase_a_trial_result_schema import (
    OPEN3D_BACKEND,
    PCL_BACKEND,
    load_json_strict,
    validate_phase_a_trial_result_strict,
)


BASELINE_COMMIT = "d33685d8515c8fc8d34354570561de3bfd7107ae"
PHASE_B_PASS_TAG = "archive/zero-perturbation-phase-b-signal-pass"
DEVELOPMENT_BRANCH = "feature/zero-perturbation-full-synthetic-development"
PRE_RUN_TAG = "archive/zero-perturbation-full-synthetic-development-pre-run"
SOURCE_REPOSITORY = Path("/home/lj/Degen-LIO")
FROZEN_MAMBA_ROOT_PREFIX = "/home/lj/.local/share/degen-lio-micromamba"
MICROMAMBA_EXECUTABLE = "/home/lj/.local/bin/micromamba"
DEVELOPMENT_RUNTIME_VERSIONS: Mapping[str, str] = {
    "matplotlib": "3.8.2",
    "numpy": "1.26.4",
    "open3d": "0.19.0+b012259",
    "python": "3.11.15",
    "scikit_learn": "1.9.0",
    "scipy": "1.11.4",
}
SCENES = (
    "GEOMETRY_RICH_ROOM",
    "LONG_CORRIDOR",
    "PARALLEL_WALLS",
    "END_FACE_TRANSITION_PRESENT",
    "END_FACE_TRANSITION_WEAK",
    "END_FACE_TRANSITION_ABSENT",
    "REPEATED_STRUCTURE",
)
GEOMETRY_SEEDS = (1850310744, 1957656152, 1334931069)
MEASUREMENT_SEEDS = (217775206, 1664898153)
REPEAT_INDICES = (0, 1, 2, 3, 4)
ALL_CONDITIONS = (
    "IDEAL_MATCHED",
    "INDEPENDENT_NOISE_FREE",
    "SCAN_NOISE_ONLY",
    "MAP_NOISE_ONLY",
    "DROPOUT_ONLY",
    "FULL_NOISE",
)
NEW_CONDITIONS = ALL_CONDITIONS[1:]
PHASE_B_OVERLAP_CONDITIONS = frozenset(
    {"INDEPENDENT_NOISE_FREE", "FULL_NOISE"}
)
CONDITION_PARAMETERS: Mapping[str, Mapping[str, Any]] = {
    "IDEAL_MATCHED": {
        "independent_sampling": False,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.0,
    },
    "INDEPENDENT_NOISE_FREE": {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.0,
    },
    "SCAN_NOISE_ONLY": {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.003,
    },
    "MAP_NOISE_ONLY": {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.001,
        "scan_dropout_fraction": 0.0,
        "scan_noise_sigma_m": 0.0,
    },
    "DROPOUT_ONLY": {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.0,
        "scan_dropout_fraction": 0.01,
        "scan_noise_sigma_m": 0.0,
    },
    "FULL_NOISE": {
        "independent_sampling": True,
        "map_dropout_fraction": 0.0,
        "map_noise_sigma_m": 0.001,
        "scan_dropout_fraction": 0.01,
        "scan_noise_sigma_m": 0.003,
    },
}

PROTOCOL_RELATIVE = Path("frozen_assets/full_synthetic_development_protocol_v1.json")
MANIFEST_RELATIVE = Path("frozen_assets/full_synthetic_development_manifest_v1.json")
SNAPSHOT_LOCK_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_snapshot_lock_v1.json"
)
NEW_SNAPSHOTS_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_new_snapshots_v1.csv"
)
NEW_TRIALS_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_new_trials_v1.csv"
)
COMBINED_SNAPSHOTS_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_combined_snapshots_v1.csv"
)
COMBINED_TRIALS_RELATIVE = Path(
    "frozen_assets/full_synthetic_development_combined_trials_v1.csv"
)
SNAPSHOT_CACHE_RELATIVE = Path("data/full_synthetic_development_v1_snapshots")
PREPARATION_REPORT_RELATIVE = Path(
    "artifacts/full_synthetic_development_snapshot_preparation.json"
)
PHASE_A_IMPORT_REPORT_RELATIVE = Path(
    "artifacts/full_synthetic_development_phase_a_ideal_import.json"
)
PHASE_B_SUBSET_REPORT_RELATIVE = Path(
    "results/full_synthetic_development_v1/phase_b_subset_reproduction.json"
)
PHASE_B_SUBSET_EVIDENCE_RELATIVE = Path(
    "artifacts/full_synthetic_phase_b_subset_reproduction_manifest.json"
)
PRE_RUN_GATE_REPORT_RELATIVE = Path(
    "artifacts/full_synthetic_development_pre_run_gate_report.json"
)

SNAPSHOT_COLUMNS = (
    "snapshot_id",
    "scene_variant",
    "geometry_seed_index",
    "geometry_seed_value",
    "measurement_seed_index",
    "measurement_seed_value",
    "repeat_index",
    "condition",
    "snapshot_origin",
    "phase_b_overlap",
)
TRIAL_COLUMNS = SNAPSHOT_COLUMNS + ("backend", "planned_trial_id")
BACKENDS = (OPEN3D_PLAN_BACKEND, PCL_PLAN_BACKEND)

REQUIRED_PRE_RUN_GATES = (
    "PHASE_A_IDEAL_IMPORT_PASS",
    "PHASE_B_PASS_ARCHIVE_VALID",
    "PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS",
    "PHASE_B_SUBSET_REPRODUCTION_PASS",
    "SCIENTIFIC_PROTOCOL_FROZEN_PASS",
    "FULL_SYNTHETIC_PLAN_PASS",
    "GENERATOR_REGRESSION_PASS",
    "FULL_SYNTHETIC_TEST_PASS",
    "NEW_1050_SNAPSHOTS_COMPLETE",
    "FULL_SYNTHETIC_DRY_RUN_PASS",
    "SOURCE_RUNTIME_ISOLATION_PASS",
    "DRY_RUN_ZERO_EXECUTION_PASS",
)

REQUIRED_PRE_SUBSET_GATES = tuple(
    name for name in REQUIRED_PRE_RUN_GATES if name != "PHASE_B_SUBSET_REPRODUCTION_PASS"
)

EXISTING_TEST_RESULTS_RELATIVE = Path(
    "artifacts/full_synthetic_existing_test_results.xml"
)
NEW_TEST_RESULTS_RELATIVE = Path("artifacts/full_synthetic_new_test_results.xml")
DRY_RUN_REPORT_RELATIVE = Path(
    "artifacts/full_synthetic_development_dry_run_report.json"
)

EXISTING_TEST_FILES = (
    "tests/test_minimal_harness.py",
    "tests/test_phase_b_analysis.py",
    "tests/test_phase_b_assets_runner.py",
    "tests/test_phase_b_frozen_integration.py",
    "tests/test_phase_b_trial_bridge.py",
)
NEW_TEST_FILES = (
    "tests/test_full_synthetic_protocol_assets_runner.py",
    "tests/test_full_synthetic_diagnostics_models.py",
    "tests/test_full_synthetic_analysis_publication.py",
)
EXISTING_TEST_COMMAND = (
    f"MAMBA_ROOT_PREFIX={FROZEN_MAMBA_ROOT_PREFIX} PYTHONNOUSERSITE=1 "
    f"{MICROMAMBA_EXECUTABLE} run "
    "-n degen-lio-zprm-py311 python -m pytest -q "
    + " ".join(EXISTING_TEST_FILES)
    + " --junitxml=artifacts/full_synthetic_existing_test_results.xml"
)
NEW_TEST_COMMAND = (
    f"MAMBA_ROOT_PREFIX={FROZEN_MAMBA_ROOT_PREFIX} PYTHONNOUSERSITE=1 "
    f"{MICROMAMBA_EXECUTABLE} run "
    "-n degen-lio-zprm-py311 python -m pytest -q "
    + " ".join(NEW_TEST_FILES)
    + " --junitxml=artifacts/full_synthetic_new_test_results.xml"
)
# Filled only after the final 65-test and 35-test collections are fixed.  The
# verifier refuses any XML while either value is not a real SHA-256.
EXISTING_TEST_COLLECTION_SHA256 = (
    "c7fd7d9cc860606065281548a0dc3a776b0fc2077d0bf030e06c7da167b0ff28"
)
NEW_TEST_COLLECTION_SHA256 = (
    "9674fad7a77002d06f346cfecd3647110fe3bcb53758185e7b011748443c133d"
)
FORMAL_EXECUTION_COMMAND = (
    f"MAMBA_ROOT_PREFIX={FROZEN_MAMBA_ROOT_PREFIX} PYTHONNOUSERSITE=1 "
    f"{MICROMAMBA_EXECUTABLE} run -n degen-lio-zprm-py311 "
    "python scripts/run_full_synthetic_development.py "
    "--manifest frozen_assets/full_synthetic_development_manifest_v1.json "
    "--run-id full-synthetic-development-v1 "
    "--output-dir results/full_synthetic_development_v1 "
    "--workers 2 --resume"
)
REQUIRED_NEW_TEST_CATEGORIES = (
    "six_condition_contract",
    "ideal_read_only_import",
    "new_snapshot_count_1050",
    "combined_snapshot_count_1260",
    "new_trial_count_2100",
    "combined_trial_count_2520",
    "phase_b_subset_reproduction",
    "snapshot_shared_by_two_backends",
    "confirmatory_seed_not_accessed",
    "native_trial_count_zero",
    "systematic_offset_definition",
    "repeatability_covariance_ddof_1",
    "systematic_fraction_zero_denominator",
    "direction_concentration",
    "common_nearest_neighbor_contract",
    "pca_normal_contract",
    "turnover_jaccard",
    "accepted_source_turnover",
    "normal_sign_invariant_angle",
    "normalized_translation_hessian",
    "spectral_entropy",
    "initial_translation_gradient",
    "leave_one_geometry_seed_out",
    "scaler_fit_on_training_fold_only",
    "fixed_ridge_alpha",
    "counterexample_pair_conditions",
    "scene_paired_win_rate",
    "cross_backend_spearman",
    "primary_analysis_verifier_agreement",
    "artifact_verifier",
)
FROZEN_GENERATOR_RUNTIME_FILES = (
    "configs/zero_perturbation/development_v1.yaml",
    "src/phase_a_harness/phase_b_generator_frozen/__init__.py",
    "src/phase_a_harness/phase_b_generator_frozen/capture_range/__init__.py",
    "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_protocol.py",
    "src/phase_a_harness/phase_b_generator_frozen/capture_range/day2_development_scene.py",
    "src/phase_a_harness/phase_b_generator_frozen/capture_range/types.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_protocol.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/backend_phase_a_v1_2.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/protocol.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/snapshot_builder.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/types.py",
    "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/__init__.py",
)


def is_phase_b_overlap_key(
    *, measurement_seed_index: int, repeat_index: int, condition: str
) -> bool:
    return bool(
        measurement_seed_index == 0
        and repeat_index == 0
        and condition in PHASE_B_OVERLAP_CONDITIONS
    )


def snapshot_id_for(
    *,
    scene: str,
    geometry_seed_index: int,
    measurement_seed_index: int,
    repeat_index: int,
    condition: str,
) -> str:
    if (
        scene not in SCENES
        or geometry_seed_index not in range(3)
        or measurement_seed_index not in range(2)
        or repeat_index not in REPEAT_INDICES
        or condition not in ALL_CONDITIONS
    ):
        raise ValueError("snapshot key is outside Full Synthetic Development v1")
    if condition == "IDEAL_MATCHED":
        return (
            f"phase-a-v1/{scene}/{geometry_seed_index}/"
            f"{measurement_seed_index}/{repeat_index}"
        )
    if is_phase_b_overlap_key(
        measurement_seed_index=measurement_seed_index,
        repeat_index=repeat_index,
        condition=condition,
    ):
        # Snapshot checksum includes snapshot_id; retaining the published ID is
        # necessary for an exact Phase B reproduction test.
        return f"phase-b-signal-v1/{scene}/g{geometry_seed_index}/{condition}"
    return (
        f"full-synthetic-development-v1/{scene}/g{geometry_seed_index}/"
        f"m{measurement_seed_index}/r{repeat_index}/{condition}"
    )


def _snapshot_row(
    *,
    scene: str,
    geometry_index: int,
    measurement_index: int,
    repeat_index: int,
    condition: str,
) -> dict[str, str]:
    overlap = is_phase_b_overlap_key(
        measurement_seed_index=measurement_index,
        repeat_index=repeat_index,
        condition=condition,
    )
    return {
        "snapshot_id": snapshot_id_for(
            scene=scene,
            geometry_seed_index=geometry_index,
            measurement_seed_index=measurement_index,
            repeat_index=repeat_index,
            condition=condition,
        ),
        "scene_variant": scene,
        "geometry_seed_index": str(geometry_index),
        "geometry_seed_value": str(GEOMETRY_SEEDS[geometry_index]),
        "measurement_seed_index": str(measurement_index),
        "measurement_seed_value": str(MEASUREMENT_SEEDS[measurement_index]),
        "repeat_index": str(repeat_index),
        "condition": condition,
        "snapshot_origin": (
            "PHASE_A_IDEAL_IMPORT"
            if condition == "IDEAL_MATCHED"
            else "NEW_DEVELOPMENT_GENERATED"
        ),
        "phase_b_overlap": "true" if overlap else "false",
    }


def planned_new_snapshots() -> list[dict[str, str]]:
    return [
        _snapshot_row(
            scene=scene,
            geometry_index=geometry_index,
            measurement_index=measurement_index,
            repeat_index=repeat_index,
            condition=condition,
        )
        for scene in SCENES
        for geometry_index in range(3)
        for measurement_index in range(2)
        for repeat_index in REPEAT_INDICES
        for condition in NEW_CONDITIONS
    ]


def planned_combined_snapshots() -> list[dict[str, str]]:
    return [
        _snapshot_row(
            scene=scene,
            geometry_index=geometry_index,
            measurement_index=measurement_index,
            repeat_index=repeat_index,
            condition=condition,
        )
        for scene in SCENES
        for geometry_index in range(3)
        for measurement_index in range(2)
        for repeat_index in REPEAT_INDICES
        for condition in ALL_CONDITIONS
    ]


def planned_trials(snapshots: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **{name: str(snapshot[name]) for name in SNAPSHOT_COLUMNS},
            "backend": backend,
            "planned_trial_id": f"{snapshot['snapshot_id']}/{backend}",
        }
        for snapshot in snapshots
        for backend in BACKENDS
    ]


def phase_b_overlap_snapshots(
    snapshots: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows = planned_new_snapshots() if snapshots is None else snapshots
    return [dict(row) for row in rows if str(row["phase_b_overlap"]) == "true"]


def validate_full_synthetic_plans(
    new_snapshots: Sequence[Mapping[str, str]],
    new_trials: Sequence[Mapping[str, str]],
    combined_snapshots: Sequence[Mapping[str, str]],
    combined_trials: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    expected_new = planned_new_snapshots()
    expected_combined = planned_combined_snapshots()
    normalized_new = [
        {name: str(row[name]) for name in SNAPSHOT_COLUMNS} for row in new_snapshots
    ]
    normalized_combined = [
        {name: str(row[name]) for name in SNAPSHOT_COLUMNS}
        for row in combined_snapshots
    ]
    normalized_new_trials = [
        {name: str(row[name]) for name in TRIAL_COLUMNS} for row in new_trials
    ]
    normalized_combined_trials = [
        {name: str(row[name]) for name in TRIAL_COLUMNS} for row in combined_trials
    ]
    if normalized_new != expected_new or normalized_combined != expected_combined:
        raise ValueError("Full Synthetic snapshot plan rows/order changed")
    if normalized_new_trials != planned_trials(expected_new):
        raise ValueError("Full Synthetic new trial plan rows/order changed")
    if normalized_combined_trials != planned_trials(expected_combined):
        raise ValueError("Full Synthetic combined trial plan rows/order changed")
    if len(normalized_new) != 1050 or len(normalized_combined) != 1260:
        raise ValueError("Full Synthetic snapshot counts changed")
    if len(normalized_new_trials) != 2100 or len(normalized_combined_trials) != 2520:
        raise ValueError("Full Synthetic trial counts changed")
    if len({row["snapshot_id"] for row in normalized_combined}) != 1260:
        raise ValueError("Full Synthetic snapshot IDs are not unique")
    if len({row["planned_trial_id"] for row in normalized_combined_trials}) != 2520:
        raise ValueError("Full Synthetic trial IDs are not unique")
    new_condition_counts = Counter(row["condition"] for row in normalized_new)
    combined_condition_counts = Counter(row["condition"] for row in normalized_combined)
    backend_counts = Counter(row["backend"] for row in normalized_combined_trials)
    pairing = Counter(row["snapshot_id"] for row in normalized_combined_trials)
    if new_condition_counts != Counter({condition: 210 for condition in NEW_CONDITIONS}):
        raise ValueError("each new Development condition must contain 210 snapshots")
    if combined_condition_counts != Counter(
        {condition: 210 for condition in ALL_CONDITIONS}
    ):
        raise ValueError("each combined Development condition must contain 210 snapshots")
    if backend_counts != Counter(
        {OPEN3D_PLAN_BACKEND: 1260, PCL_PLAN_BACKEND: 1260}
    ) or set(pairing.values()) != {2}:
        raise ValueError("Full Synthetic dual-backend sharing changed")
    overlap = phase_b_overlap_snapshots(normalized_new)
    if len(overlap) != 42 or len(planned_trials(overlap)) != 84:
        raise ValueError("published Phase B overlap must remain 42/84")
    return {
        "COMBINED_1260_SNAPSHOTS_PLANNED": True,
        "COMBINED_2520_TRIALS_PLANNED": True,
        "FULL_SYNTHETIC_PLAN_PASS": True,
        "NEW_1050_SNAPSHOTS_PLANNED": True,
        "NEW_2100_TRIALS_PLANNED": True,
        "backend_trial_counts": dict(sorted(backend_counts.items())),
        "combined_condition_snapshot_counts": dict(
            sorted(combined_condition_counts.items())
        ),
        "combined_snapshot_count": len(normalized_combined),
        "combined_trial_count": len(normalized_combined_trials),
        "native_trial_count": 0,
        "new_condition_snapshot_counts": dict(sorted(new_condition_counts.items())),
        "new_snapshot_count": len(normalized_new),
        "new_trial_count": len(normalized_new_trials),
        "phase_b_overlap_snapshot_count": len(overlap),
        "phase_b_overlap_trial_count": len(planned_trials(overlap)),
        "snapshot_backend_pairing_mismatch_count": 0,
    }


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: str(row[name]) for name in columns})
    return stream.getvalue().encode("utf-8")


def _write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"existing frozen asset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_full_synthetic_plans(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    new_snapshots = planned_new_snapshots()
    new_trials = planned_trials(new_snapshots)
    combined_snapshots = planned_combined_snapshots()
    combined_trials = planned_trials(combined_snapshots)
    report = validate_full_synthetic_plans(
        new_snapshots, new_trials, combined_snapshots, combined_trials
    )
    bindings = (
        (NEW_SNAPSHOTS_RELATIVE, SNAPSHOT_COLUMNS, new_snapshots),
        (NEW_TRIALS_RELATIVE, TRIAL_COLUMNS, new_trials),
        (COMBINED_SNAPSHOTS_RELATIVE, SNAPSHOT_COLUMNS, combined_snapshots),
        (COMBINED_TRIALS_RELATIVE, TRIAL_COLUMNS, combined_trials),
    )
    for relative, columns, rows in bindings:
        _write_once_or_verify(repository / relative, _csv_bytes(columns, rows))
    return {
        **report,
        "files": {
            relative.as_posix(): file_sha256(repository / relative)
            for relative, _, _ in bindings
        },
    }


def read_full_synthetic_plans(
    root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    repository = Path(root).resolve()

    def rows(relative: Path, columns: Sequence[str]) -> list[dict[str, str]]:
        with (repository / relative).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ValueError(f"Full Synthetic plan columns changed: {relative}")
            return list(reader)

    result = (
        rows(NEW_SNAPSHOTS_RELATIVE, SNAPSHOT_COLUMNS),
        rows(NEW_TRIALS_RELATIVE, TRIAL_COLUMNS),
        rows(COMBINED_SNAPSHOTS_RELATIVE, SNAPSHOT_COLUMNS),
        rows(COMBINED_TRIALS_RELATIVE, TRIAL_COLUMNS),
    )
    validate_full_synthetic_plans(*result)
    return result


def _verify_sha256s(directory: Path) -> tuple[int, list[str]]:
    sums = directory / "SHA256SUMS"
    mismatches: list[str] = []
    entries = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        candidate = (directory / relative).resolve()
        if directory.resolve() not in candidate.parents or not candidate.is_file():
            mismatches.append(relative)
        elif file_sha256(candidate) != digest:
            mismatches.append(relative)
        entries += 1
    return entries, mismatches


def _load_json_object_strict(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key in {label}: {name}")
            result[name] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {label}: {token}")
        ),
    )
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def verify_phase_a_ideal_import(
    root: str | Path, *, write_report: bool = True
) -> dict[str, Any]:
    """Validate and expose Phase A IDEAL results without copying or modifying them."""

    repository = Path(root).resolve()
    from .asset_verifier import verify_frozen_assets

    frozen_report = verify_frozen_assets(
        repository / "frozen_assets/frozen_experiment_manifest.json",
        write_report=False,
    )
    artifact_root = repository / "artifacts/formal_phase_a_v1"
    artifact_verification = _load_json_object_strict(
        artifact_root / "artifact_verification.json", "Phase A artifact verification"
    )
    final_decision = _load_json_object_strict(
        artifact_root / "final_decision.json", "Phase A final decision"
    )
    sha_entries, artifact_mismatches = _verify_sha256s(artifact_root)

    run_manifest_path = repository / "results/formal_phase_a_v1/run_manifest.json"
    run_root = run_manifest_path.parent.resolve()  # authoritative actual result path
    run_manifest = _load_json_object_strict(run_manifest_path, "Phase A run manifest")
    raw_manifest_path = run_root / "raw_result_manifest.json"
    raw_manifest = _load_json_object_strict(
        raw_manifest_path, "Phase A raw result manifest"
    )
    if (
        raw_manifest.get("run_id") != "phase-a-minimal-harness-formal-v1"
        or type(raw_manifest.get("results")) is not dict
    ):
        raise ValueError("Phase A raw result manifest identity changed")

    ideal_snapshots = [
        {name: row[name] for name in SNAPSHOT_COLUMNS[:8]}
        for row in planned_combined_snapshots()
        if row["condition"] == "IDEAL_MATCHED"
    ]
    ideal_trials = [
        {name: row[name] for name in TRIAL_COLUMNS[:8] + ("backend", "planned_trial_id")}
        for row in planned_trials(
            [row for row in planned_combined_snapshots() if row["condition"] == "IDEAL_MATCHED"]
        )
    ]

    def read_plan(relative: str, columns: Sequence[str]) -> list[dict[str, str]]:
        with (repository / relative).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ValueError(f"Phase A plan columns changed: {relative}")
            return list(reader)

    imported_snapshot_plan = read_plan(
        "frozen_assets/planned_snapshots.csv", SNAPSHOT_COLUMNS[:8]
    )
    imported_trial_columns = SNAPSHOT_COLUMNS[:8] + ("backend", "planned_trial_id")
    imported_trial_plan = read_plan(
        "frozen_assets/planned_trials.csv", imported_trial_columns
    )
    snapshot_plan_mismatch = imported_snapshot_plan != ideal_snapshots
    trial_plan_mismatch = imported_trial_plan != ideal_trials
    expected_snapshot_by_id = {row["snapshot_id"]: row for row in ideal_snapshots}
    expected_trial_by_id = {row["planned_trial_id"]: row for row in ideal_trials}
    expected_trial_ids = set(expected_trial_by_id)
    raw_trial_ids = set(raw_manifest["results"])
    raw_trial_id_mismatch_count = len(expected_trial_ids ^ raw_trial_ids)

    result_entries = raw_manifest["results"]
    manifest_paths = [
        entry.get("path")
        for entry in result_entries.values()
        if type(entry) is dict
    ]
    result_path_duplicate_count = len(manifest_paths) - len(set(manifest_paths))
    raw_results_root = run_root / "raw_results"
    actual_raw_files = {
        path.name for path in raw_results_root.iterdir() if path.is_file()
    }
    expected_raw_files = {path for path in manifest_paths if type(path) is str}
    reverse_inventory_mismatch_count = len(actual_raw_files ^ expected_raw_files)

    snapshot_metadata_mismatch_count = 0
    snapshot_metadata_by_id: dict[str, dict[str, Any]] = {}
    for snapshot_id, expected in expected_snapshot_by_id.items():
        metadata_path = repository / "data/frozen_snapshots" / snapshot_id / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            exact = {
                "condition": expected["condition"],
                "geometry_seed": int(expected["geometry_seed_value"]),
                "geometry_seed_index": int(expected["geometry_seed_index"]),
                "measurement_seed": int(expected["measurement_seed_value"]),
                "measurement_seed_index": int(expected["measurement_seed_index"]),
                "repeat_index": int(expected["repeat_index"]),
                "scene_variant": expected["scene_variant"],
                "snapshot_id": snapshot_id,
            }
            if any(metadata.get(name) != value for name, value in exact.items()):
                raise ValueError("Phase A snapshot metadata identity mismatch")
            snapshot_metadata_by_id[snapshot_id] = metadata
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            snapshot_metadata_mismatch_count += 1

    payloads: list[dict[str, Any]] = []
    corrupt = 0
    trial_identity_mismatch_count = 0
    for trial_id, entry in sorted(raw_manifest["results"].items()):
        try:
            if type(entry) is not dict or set(entry) != {
                "path",
                "planned_trial_id",
                "sha256",
            }:
                raise ValueError("Phase A raw result manifest entry schema changed")
            path = run_root / "raw_results" / entry["path"]
            if (
                entry.get("planned_trial_id") != trial_id
                or Path(entry["path"]).name != entry["path"]
                or file_sha256(path) != entry.get("sha256")
            ):
                raise ValueError("Phase A raw result manifest binding mismatch")
            payload = validate_phase_a_trial_result_strict(load_json_strict(path))
            expected = expected_trial_by_id.get(trial_id)
            metadata = snapshot_metadata_by_id.get(payload["snapshot_id"])
            expected_backend = (
                OPEN3D_BACKEND
                if expected and expected["backend"] == OPEN3D_PLAN_BACKEND
                else PCL_BACKEND
            )
            if (
                expected is None
                or metadata is None
                or payload["planned_trial_id"] != trial_id
                or payload["snapshot_id"] != expected["snapshot_id"]
                or payload["scene_variant"] != expected["scene_variant"]
                or payload["condition"] != "IDEAL_MATCHED"
                or payload["backend"] != expected_backend
                or payload["snapshot_checksum"] != metadata["snapshot_checksum"]
                or payload["source_checksum"] != metadata["source_raw_checksum"]
                or payload["target_checksum"] != metadata["target_raw_checksum"]
                or payload["reference_pose_checksum"]
                != metadata["reference_pose_raw_checksum"]
            ):
                trial_identity_mismatch_count += 1
                raise ValueError("Phase A raw trial identity mismatch")
            payloads.append(payload)
        except (KeyError, OSError, ValueError):
            corrupt += 1
    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        by_snapshot[payload["snapshot_id"]].append(payload)
    pairing_mismatch = 0
    checksum_fields = (
        "snapshot_checksum",
        "source_checksum",
        "target_checksum",
        "reference_pose_checksum",
    )
    for rows in by_snapshot.values():
        if len(rows) != 2 or {row["backend"] for row in rows} != {
            OPEN3D_BACKEND,
            PCL_BACKEND,
        }:
            pairing_mismatch += 1
        elif any(rows[0][name] != rows[1][name] for name in checksum_fields):
            pairing_mismatch += 1
    backend_counts = Counter(row["backend"] for row in payloads)
    solver_failure_count = sum(bool(row["solver_failure"]) for row in payloads)
    nonfinite_count = sum(not bool(row["finite_output"]) for row in payloads)
    pass_value = bool(
        frozen_report.get("SCIENTIFIC_ASSET_EXPORT_EQUIVALENCE_PASS") is True
        and artifact_verification.get("ARTIFACT_VERIFICATION_PASS") is True
        and final_decision.get("DAY1_SCIENTIFIC_VALIDATION_PASS") is True
        and final_decision.get("TWO_INDEPENDENT_BACKENDS_QUALIFIED") is True
        and final_decision.get("BACKEND_PHASE_A_COMPLETE") is True
        and not snapshot_plan_mismatch
        and not trial_plan_mismatch
        and raw_trial_id_mismatch_count == 0
        and result_path_duplicate_count == 0
        and reverse_inventory_mismatch_count == 0
        and snapshot_metadata_mismatch_count == 0
        and trial_identity_mismatch_count == 0
        and len(payloads) == 420
        and len(by_snapshot) == 210
        and backend_counts == Counter({OPEN3D_BACKEND: 210, PCL_BACKEND: 210})
        and run_manifest.get("completed_snapshot_count") == 210
        and run_manifest.get("completed_trial_count") == 420
        and run_manifest.get("open3d_trial_count") == 210
        and run_manifest.get("pcl_trial_count") == 210
        and run_manifest.get("native_execution_count") == 0
        and corrupt == 0
        and pairing_mismatch == 0
        and solver_failure_count == 0
        and nonfinite_count == 0
        and not artifact_mismatches
    )
    report = {
        "PHASE_A_IDEAL_IMPORT_PASS": pass_value,
        "artifact_sha256_entry_count": sha_entries,
        "artifact_sha256_mismatch_count": len(artifact_mismatches),
        "backend_input_checksum_mismatch_count": pairing_mismatch,
        "completed_snapshot_count": len(by_snapshot),
        "completed_trial_count": len(payloads),
        "corrupt_trial_count": corrupt,
        "expected_trial_id_mismatch_count": raw_trial_id_mismatch_count,
        "native_trial_count": 0,
        "nonfinite_output_count": nonfinite_count,
        "open3d_trial_count": backend_counts[OPEN3D_BACKEND],
        "pcl_trial_count": backend_counts[PCL_BACKEND],
        "phase_a_artifact_root": artifact_root.relative_to(repository).as_posix(),
        "phase_a_raw_result_manifest_path": raw_manifest_path.relative_to(repository).as_posix(),
        "phase_a_results_root": run_root.relative_to(repository).as_posix(),
        "phase_a_run_manifest_path": run_manifest_path.relative_to(repository).as_posix(),
        "result_path_duplicate_count": result_path_duplicate_count,
        "reverse_raw_result_inventory_mismatch_count": reverse_inventory_mismatch_count,
        "schema_version": "full_synthetic_phase_a_ideal_import_v1",
        "snapshot_metadata_identity_mismatch_count": snapshot_metadata_mismatch_count,
        "snapshot_plan_mismatch_count": int(snapshot_plan_mismatch),
        "solver_failure_count": solver_failure_count,
        "trial_identity_mismatch_count": trial_identity_mismatch_count,
        "trial_plan_mismatch_count": int(trial_plan_mismatch),
    }
    if write_report:
        write_json(repository / PHASE_A_IMPORT_REPORT_RELATIVE, report)
    return report


def full_synthetic_protocol_payload() -> dict[str, Any]:
    return {
        "authorization_ceiling": {
            "CONFIRMATORY_RUN_AUTHORIZED": False,
            "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
            "REAL_DATA_RUN_AUTHORIZED": False,
        },
        "backend_parameters": {
            "contract_path": "frozen_assets/backend_parameter_contract.json",
            "open3d_sha256": "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413",
            "pcl_sha256": "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd",
        },
        "bootstrap": {
            "outer_unit": "geometry_seed",
            "inner_unit": "measurement_seed_x_repeat",
            "repetitions": 2000,
            "seed": 1191248828,
            "scope": "exploratory_development_only",
        },
        "common_association": {
            "invalid_reasons": [
                "NO_INITIAL_CORRESPONDENCE",
                "NO_FINAL_CORRESPONDENCE",
                "INSUFFICIENT_VALID_NORMALS",
                "NONFINITE_COMMON_METRICS",
                "OTHER",
            ],
            "maximum_correspondence_distance_m": 0.5,
            "minimum_valid_fraction": 0.95,
            "normal_estimation": {
                "covariance_denominator": "k",
                "eigensolver": "numpy.linalg.eigh",
                "k": 50,
                "minimum_neighbors": 10,
                "orientation_unified": False,
            },
            "target_index": "scipy.spatial.cKDTree",
            "turnover": "one_minus_pair_set_jaccard",
        },
        "conditions": {
            name: dict(CONDITION_PARAMETERS[name]) for name in ALL_CONDITIONS
        },
        "counts": {
            "combined_snapshots": 1260,
            "combined_trials": 2520,
            "new_open3d_trials": 1050,
            "new_pcl_trials": 1050,
            "new_snapshots": 1050,
            "new_trials": 2100,
            "native_trials": 0,
            "phase_a_ideal_snapshots": 210,
            "phase_a_ideal_trials": 420,
            "phase_b_overlap_snapshots": 42,
            "phase_b_overlap_trials": 84,
        },
        "cross_validation": {
            "folds": 3,
            "ridge_alpha": 1.0,
            "ridge_fit_intercept": True,
            "split": "leave_one_geometry_seed_out",
            "standard_scaler_fit": "training_fold_only",
            "target": "log10_translation_error_plus_1e-9",
        },
        "execution_environment": {
            "development_runtime_versions": dict(DEVELOPMENT_RUNTIME_VERSIONS),
            "environment_manifest_path": "frozen_assets/environment_manifest.json",
            "hard_gates": {
                "mamba_root_prefix": FROZEN_MAMBA_ROOT_PREFIX,
                "micromamba_executable": MICROMAMBA_EXECUTABLE,
                "open3d_version": "0.19.0+b012259",
                "pcl_cli_sha256": "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff",
                "pcl_linked_abi": "1.15",
                "pcl_version": "1.15.1",
            },
            "python_no_user_site_required": True,
            "source_repository_python_path_forbidden": True,
        },
        "formal_execution": {
            "command": FORMAL_EXECUTION_COMMAND,
            "command_sha256": canonical_json_sha256(
                {"command": FORMAL_EXECUTION_COMMAND}
            ),
            "output_directory": "results/full_synthetic_development_v1",
            "resume_required": True,
            "run_id": "full-synthetic-development-v1",
            "workers": 2,
        },
        "gates": {
            "automatic_nonequivalence_candidate": {
                "absolute_log_condition_ratio_max": 0.20,
                "absolute_log_initial_rmse_ratio_max": 0.20,
                "allowed_manual_review_status": "AUTOMATIC_CANDIDATE",
                "conditions": list(NEW_CONDITIONS),
                "correspondence_count_ratio_max": 1.10,
                "correspondence_count_ratio_min": 0.90,
                "finite_translation_error_required_both": True,
                "finite_turnover_required_both": True,
                "forbidden_automatic_status": "VALID_SCIENTIFIC_COUNTEREXAMPLE",
                "normalized_hessian_eigenvalue_cosine_similarity_min": 0.98,
                "search_scope": "within_backend_and_condition_snapshot_pairs",
                "translation_error_ratio_min": 5.0,
                "turnover_absolute_difference_min": 0.15,
            },
            "common_association_valid_fraction_min": 0.95,
            "cross_backend": {
                "independent_noise_free_rho_min": 0.70,
                "full_noise_rho_min": 0.70,
                "at_least_four_of_five_rho_min": 0.50,
                "median_five_rho_min": 0.70,
                "pooled_rho_min": 0.75,
            },
            "execution_success": {
                "backend_condition_min": 0.95,
                "backend_condition_scene_min": 0.90,
            },
            "local_metric_incremental_value": {
                "mean_relative_mae_improvement_min": 0.10,
                "per_backend_relative_improvement_min": -0.02,
                "alternative_candidate_count_per_backend_min": 10,
                "alternative_scene_pair_coverage_min": 3,
            },
            "primary_scene_effect": {
                "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
                "control": "GEOMETRY_RICH_ROOM",
                "minimum_absolute_difference_m": 0.005,
                "minimum_paired_wins": 24,
                "minimum_ratio": 5.0,
                "minimum_success_rate": 0.90,
                "scene": "LONG_CORRIDOR",
                "matched_blocks": 30,
            },
            "reassociation": {
                "pooled_rho_min_each_backend": 0.40,
                "centered_rho_min_one_backend": 0.20,
                "centered_rho_min_other_backend": 0.0,
            },
            "scene_rank_stability": {
                "condition_backend_combination_count": 10,
                "conditions": list(NEW_CONDITIONS),
                "geometry_rich_room_low_error_rank_max": 2,
                "geometry_rich_room_required_combinations_min": 8,
                "weak_scene_candidates": [
                    "LONG_CORRIDOR",
                    "END_FACE_TRANSITION_ABSENT",
                ],
                "weak_scene_high_error_rank_max": 3,
                "weak_scene_required_combinations_min": 8,
                "weak_scene_rule": "at_least_one_candidate_within_highest_error_rank",
            },
            "systematic_offset_claim": {
                "all_backend_condition_combinations_required": True,
                "backend_condition_combination_count": 4,
                "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
                "geometry_groups_per_backend_condition": 3,
                "group_observation_count": 10,
                "group_systematic_fraction_translation_min": 0.60,
                "group_systematic_translation_offset_m_min": 0.005,
                "median_systematic_fraction_translation_min": 0.70,
                "qualified_geometry_groups_min": 2,
                "scene": "LONG_CORRIDOR",
                "wording_gate_only_not_development_pass_requirement": True,
            },
        },
        "generator_contract": {
            "generator_sha256": "f1632095ab6c433e1e917cf0c9a49551683fce2f106761d41c3c3a5e6517b922",
            "method_name_in_seed": False,
            "shared_snapshot_across_backends": True,
            "snapshot_builder_sha256": "6718fc442439e52327e01622df6356a0167456a6d7d8070f25d6e684ef69b72e",
        },
        "geometry_seeds": list(GEOMETRY_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "metric_contract": {
            "initial_transform": "reference_pose_exact",
            "relative_transform": "inverse(T_reference) @ T_estimated",
            "rotation": "norm(Log_SO3(nearest_SO3_reflection_safe(R_reference.T @ R_estimated)))",
            "rotation_geodesic": "atan2_geodesic_not_raw_trace_acos",
            "rotation_vector": "Log_SO3(R_reference.T @ R_estimated)",
            "translation": "norm(T_estimated[0:3,3] - T_reference[0:3,3])",
            "translation_vector": "T_estimated[0:3,3] - T_reference[0:3,3]",
        },
        "statistical_reporting": {
            "development_intervals_are_confirmatory": False,
            "iqr": "q75_minus_q25",
            "median_ratio": "median(numerator) / median(denominator)",
            "paired_median_difference": "median(blockwise_value_a_minus_value_b)",
            "paired_win_rate": "count(blockwise_value_a_gt_value_b) / matched_block_count",
            "quantile_implementation": "numpy.quantile",
            "quantile_method": "linear",
            "reported": [
                "median",
                "IQR",
                "q95",
                "paired_median_difference",
                "median_ratio",
                "exploratory_95_percent_bootstrap_interval",
                "paired_win_rate",
                "spearman_rho",
                "spearman_exploratory_interval",
            ],
            "spearman": "scipy.stats.spearmanr",
        },
        "systematic_offset_and_repeatability": {
            "direction_concentration": "norm(sum(rho_k / norm(rho_k))) / valid_nonzero_count",
            "direction_concentration_excludes_zero_vectors": True,
            "group_keys": [
                "scene_variant",
                "geometry_seed",
                "condition",
                "backend",
            ],
            "group_observations": "2_measurement_seeds_x_5_repeats_equals_10",
            "mean_rotation_vector": "mean(phi_k)",
            "mean_translation_vector": "mean(rho_k)",
            "rotation_repeatability_covariance": "cov(phi_k, ddof=1)",
            "rotation_repeatability_rms_rad": "sqrt(trace(rotation_repeatability_covariance))",
            "systematic_fraction_translation": "norm(mean_translation_vector) / mean(norm(rho_k))",
            "systematic_fraction_zero_denominator_epsilon": 1e-12,
            "systematic_fraction_zero_denominator_value": None,
            "systematic_rotation_offset_rad": "norm(mean_rotation_vector)",
            "systematic_translation_offset_m": "norm(mean_translation_vector)",
            "terminology_requires_systematic_offset_claim_authorized": True,
            "translation_repeatability_covariance": "cov(rho_k, ddof=1)",
            "translation_repeatability_rms_m": "sqrt(trace(translation_repeatability_covariance))",
        },
        "model_features": {
            "model_a": [
                "log10_initial_residual_rmse_plus_1e-9",
                "log10_condition_number_trans_plus_1",
                "log10_inverse_lambda_min_trans",
                "spectral_entropy_trans",
                "log10_initial_correspondence_count_plus_1",
                "initial_translation_gradient_norm",
            ],
            "model_b_additions": [
                "correspondence_turnover",
                "accepted_source_turnover",
                "median_normal_angle_change_deg",
                "q95_normal_angle_change_deg",
                "residual_rmse_change",
                "correspondence_count_change_ratio",
            ],
        },
        "phase_a_import": {
            "artifact_root": "artifacts/formal_phase_a_v1",
            "formal_run_manifest_path": "results/formal_phase_a_v1/run_manifest.json",
            "mode": "read_only_no_copy_no_rerun",
        },
        "phase_b_reproduction": {
            "absolute_tolerance": 1e-12,
            "reference_raw_manifest_path": "results/phase_b_signal_v1/raw_result_manifest.json",
            "relative_tolerance": 1e-12,
            "subset": {
                "conditions": ["INDEPENDENT_NOISE_FREE", "FULL_NOISE"],
                "measurement_seed": 217775206,
                "repeat_index": 0,
            },
        },
        "pre_run_freeze": {
            "branch": DEVELOPMENT_BRANCH,
            "commit_message": "chore: freeze full synthetic development experiment",
            "formal_operational_sequence": [
                "authorize_manifest_from_raw_evidence",
                "commit_authorized_manifest_and_all_pre_run_evidence",
                "create_pre_run_tag_at_HEAD",
                "require_tag_equals_HEAD_and_clean_worktree_before_formal_execution",
            ],
            "required_tag": PRE_RUN_TAG,
            "self_referential_commit_hash_in_manifest": False,
        },
        "repeat_indices": list(REPEAT_INDICES),
        "scenes": list(SCENES),
        "schema_version": "full_synthetic_development_protocol_v1",
        "scientific_scope": "development_not_confirmatory",
        "source_baseline_commit": BASELINE_COMMIT,
        "test_execution": {
            "existing": {
                "collection_sha256": EXISTING_TEST_COLLECTION_SHA256,
                "command": EXISTING_TEST_COMMAND,
                "command_sha256": canonical_json_sha256(
                    {"command": EXISTING_TEST_COMMAND}
                ),
                "exact_count": 65,
                "files": list(EXISTING_TEST_FILES),
            },
            "new": {
                "collection_sha256": NEW_TEST_COLLECTION_SHA256,
                "command": NEW_TEST_COMMAND,
                "command_sha256": canonical_json_sha256(
                    {"command": NEW_TEST_COMMAND}
                ),
                "exact_count": 35,
                "files": list(NEW_TEST_FILES),
                "permitted_count_range": [25, 35],
                "required_categories": list(REQUIRED_NEW_TEST_CATEGORIES),
            },
        },
        "trial_result_schema": {
            "path": "frozen_assets/trial_result_schema.json",
            "schema_version": "phase_a_trial_result_v1",
            "sha256": "4d1b9da79d11b36b19350bec1c0b9eb84fc64a3699be682a4eecca684b451773",
        },
    }


def verify_full_synthetic_protocol_file(root: str | Path) -> dict[str, Any]:
    """Require the prospectively written protocol to match code byte-for-byte."""

    repository = Path(root).resolve()
    path = repository / PROTOCOL_RELATIVE
    expected = full_synthetic_protocol_payload()
    expected_bytes = (
        json.dumps(
            expected,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        actual_bytes = path.read_bytes()
        actual = json.loads(
            actual_bytes.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite protocol constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Full Synthetic protocol is missing or invalid") from error
    semantic = actual == expected
    byte_exact = actual_bytes == expected_bytes
    if not semantic or not byte_exact:
        raise ValueError("Full Synthetic protocol was not frozen before snapshot creation")
    return {
        "SCIENTIFIC_PROTOCOL_FROZEN_PASS": True,
        "protocol_byte_equivalence": byte_exact,
        "protocol_semantic_equivalence": semantic,
        "protocol_sha256": file_sha256(path),
    }


def assert_isolated_python_runtime(
    *,
    environ: Mapping[str, str] | None = None,
    search_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail before execution if Python can resolve modules from Degen-LIO."""

    environment = os.environ if environ is None else environ
    if environment.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("formal runtime requires PYTHONNOUSERSITE=1")
    if environment.get("MAMBA_ROOT_PREFIX") != FROZEN_MAMBA_ROOT_PREFIX:
        raise PermissionError(
            f"formal runtime requires MAMBA_ROOT_PREFIX={FROZEN_MAMBA_ROOT_PREFIX}"
        )
    entries = list(sys.path if search_paths is None else search_paths)
    entries.extend(
        item
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item
    )
    source = SOURCE_REPOSITORY.resolve()
    forbidden: list[str] = []
    for entry in entries:
        try:
            candidate = (Path.cwd() if not entry else Path(entry)).resolve()
        except (OSError, TypeError):
            continue
        if candidate == source or source in candidate.parents:
            forbidden.append(str(candidate))
    imported = sorted(
        {
            str(Path(module_file).resolve())
            for module in tuple(sys.modules.values())
            for module_file in (getattr(module, "__file__", None),)
            if isinstance(module_file, str)
            and (
                Path(module_file).resolve() == source
                or source in Path(module_file).resolve().parents
            )
        }
    )
    if forbidden or imported:
        raise PermissionError("formal Python runtime resolves to the source repository")
    return {
        "PYTHON_RUNTIME_ISOLATION_PASS": True,
        "forbidden_search_path_count": 0,
        "source_repository_runtime_import_count": 0,
    }


def verify_frozen_runtime_environment(root: str | Path) -> dict[str, Any]:
    """Read-only backend/runtime identity gate; no registration is executed."""

    repository = Path(root).resolve()
    environment_path = repository / "frozen_assets/environment_manifest.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    python_contract = environment["python"]
    pcl_contract = environment["pcl"]
    observed_python = {
        "environment": str(Path(sys.prefix).resolve()),
        "numpy": importlib.metadata.version("numpy"),
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "scipy": importlib.metadata.version("scipy"),
    }
    open3d = importlib.import_module("open3d")
    observed_python["open3d"] = str(open3d.__version__)
    try:
        observed_sklearn = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        observed_sklearn = None
    try:
        observed_matplotlib = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError:
        observed_matplotlib = None
    observed_development_versions = {
        name: observed_python[name]
        for name in ("numpy", "open3d", "python", "scipy")
    } | {
        "matplotlib": observed_matplotlib,
        "scikit_learn": observed_sklearn,
    }
    cli = repository / str(pcl_contract["cli_path"])
    ldd = subprocess.run(
        ["ldd", str(cli)],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    linked = tuple(str(name) for name in pcl_contract["linked_pcl_libraries"])
    pcl_linkage_pass = bool(
        ldd.returncode == 0
        and all(name in ldd.stdout for name in linked)
        and "not found" not in ldd.stdout
    )
    phase_a_python_manifest_pass = all(
        observed_python[name] == str(python_contract[name])
        for name in ("environment", "numpy", "open3d", "python", "scipy")
    )
    development_versions_pass = all(
        observed_development_versions[name] == expected
        for name, expected in DEVELOPMENT_RUNTIME_VERSIONS.items()
    )
    pcl_pass = bool(
        file_sha256(cli) == pcl_contract["cli_sha256"]
        and pcl_contract["version"] == "1.15.1"
        and pcl_linkage_pass
    )
    mamba_root_prefix_pass = (
        os.environ.get("MAMBA_ROOT_PREFIX") == FROZEN_MAMBA_ROOT_PREFIX
    )
    passed = bool(
        phase_a_python_manifest_pass
        and development_versions_pass
        and pcl_pass
        and mamba_root_prefix_pass
    )
    report = {
        "FROZEN_RUNTIME_ENVIRONMENT_PASS": passed,
        "environment_manifest_sha256": file_sha256(environment_path),
        "development_runtime_versions": observed_development_versions,
        "development_runtime_versions_hard_gate_pass": development_versions_pass,
        "mamba_root_prefix_hard_gate_pass": mamba_root_prefix_pass,
        "observed_mamba_root_prefix": os.environ.get("MAMBA_ROOT_PREFIX"),
        "observed_matplotlib": observed_matplotlib,
        "observed_python": observed_python,
        "observed_scikit_learn": observed_sklearn,
        "open3d_version_hard_gate_pass": observed_python["open3d"]
        == python_contract["open3d"],
        "pcl_cli_sha256_hard_gate_pass": file_sha256(cli)
        == pcl_contract["cli_sha256"],
        "pcl_linkage_hard_gate_pass": pcl_linkage_pass,
        "pcl_version_hard_gate_pass": pcl_contract["version"] == "1.15.1",
        "phase_a_environment_manifest_core_gate_pass": phase_a_python_manifest_pass,
        "phase_a_environment_manifest_matplotlib_matches": observed_matplotlib
        == python_contract.get("matplotlib"),
        "python_core_hard_gate_pass": all(
            observed_development_versions[name]
            == DEVELOPMENT_RUNTIME_VERSIONS[name]
            for name in ("numpy", "open3d", "python", "scipy")
        ),
        "scikit_learn_version_hard_gate_pass": observed_sklearn
        == DEVELOPMENT_RUNTIME_VERSIONS["scikit_learn"],
        "matplotlib_version_hard_gate_pass": observed_matplotlib
        == DEVELOPMENT_RUNTIME_VERSIONS["matplotlib"],
    }
    if not passed:
        raise PermissionError("frozen Open3D/PCL/Python runtime environment changed")
    return report


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_phase_b_pass_archive(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    artifact_root = repository / "artifacts/phase_b_signal_v1"
    final = _load_json_object_strict(
        artifact_root / "final_decision.json", "Phase B final decision"
    )
    verification = _load_json_object_strict(
        artifact_root / "artifact_verification.json",
        "Phase B artifact verification",
    )
    required_true = (
        "PHASE_B_ENGINEERING_PASS",
        "PHASE_B_WEAK_RICH_EFFECT_PASS",
        "PHASE_B_SCENE_RANKING_PASS",
        "PHASE_B_GEOMETRY_SEED_CONSISTENCY_PASS",
        "PHASE_B_CROSS_BACKEND_RANKING_PASS",
        "PHASE_B_SIGNAL_PASS",
        "FULL_SYNTHETIC_DEVELOPMENT_PROTOCOL_DESIGN_AUTHORIZED",
    )
    required_false = (
        "CONFIRMATORY_AUTHORIZED",
        "FULL_SYNTHETIC_DEVELOPMENT_RUN_AUTHORIZED",
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED",
        "REAL_DATA_AUTHORIZED",
    )
    sha_entries, mismatches = _verify_sha256s(artifact_root)
    tag_commit = _git(repository, "rev-parse", f"{PHASE_B_PASS_TAG}^{{commit}}")
    pass_value = bool(
        set(final) == set(required_true) | set(required_false)
        and all(final.get(name) is True for name in required_true)
        and all(final.get(name) is False for name in required_false)
        and verification.get("ARTIFACT_VERIFICATION_PASS") is True
        and verification.get("sha256_mismatch_count") == 0
        and verification.get("sha256_missing_count") == 0
        and not mismatches
        and tag_commit == BASELINE_COMMIT
    )
    return {
        "PHASE_B_PASS_ARCHIVE_VALID": pass_value,
        "artifact_sha256_entry_count": sha_entries,
        "artifact_sha256_mismatch_count": len(mismatches),
        "phase_b_pass_tag": PHASE_B_PASS_TAG,
        "phase_b_pass_tag_commit": tag_commit,
        "required_false_decision_field_count": len(required_false),
        "required_true_decision_field_count": len(required_true),
    }


def verify_formal_operational_git_gate(root: str | Path) -> dict[str, Any]:
    """Post-authorization gate: authorize, commit, tag, then execute cleanly.

    The manifest deliberately binds the required tag name instead of a future
    commit hash.  This removes the self-referential commit/manifest cycle while
    still requiring the tagged HEAD blob and worktree to be byte-identical.
    """

    repository = Path(root).resolve()
    top = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repository, "rev-parse", "HEAD^{commit}")
    tag_head = _git(repository, "rev-parse", f"{PRE_RUN_TAG}^{{commit}}")
    branch = _git(repository, "branch", "--show-current")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    manifest_relative = MANIFEST_RELATIVE.as_posix()
    _git(repository, "ls-files", "--error-unmatch", manifest_relative)
    committed_manifest = subprocess.run(
        ["git", "show", f"HEAD:{manifest_relative}"],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    manifest_path = repository / MANIFEST_RELATIVE
    manifest_blob_match = bool(
        committed_manifest.returncode == 0
        and committed_manifest.stdout == manifest_path.read_bytes()
    )
    pass_value = bool(
        top == repository
        and branch == DEVELOPMENT_BRANCH
        and head == tag_head
        and not status
        and manifest_blob_match
    )
    report = {
        "FORMAL_OPERATIONAL_GIT_GATE_PASS": pass_value,
        "branch": branch,
        "head_commit": head,
        "manifest_head_blob_match": manifest_blob_match,
        "pre_run_tag": PRE_RUN_TAG,
        "pre_run_tag_commit": tag_head,
        "repository_root": str(top),
        "worktree_clean": not status,
    }
    if not pass_value:
        raise PermissionError("formal pre-run tag/HEAD/clean operational gate failed")
    return report


def write_full_synthetic_protocol(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    payload = full_synthetic_protocol_payload()
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_once_or_verify(repository / PROTOCOL_RELATIVE, encoded)
    return payload


def _sha(root: Path, relative: str) -> str:
    candidate = root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"Development manifest binding missing: {relative}")
    return file_sha256(candidate)


def build_full_synthetic_manifest_payload(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    verify_full_synthetic_protocol_file(repository)
    protocol = json.loads((repository / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    code_paths = {
        "analysis": "src/phase_a_harness/full_synthetic_analysis.py",
        "artifact_verifier": "src/phase_a_harness/full_synthetic_artifact_verifier.py",
        "asset_verifier": "src/phase_a_harness/asset_verifier.py",
        "association_analysis": "src/phase_a_harness/common_association_analysis.py",
        "backend_metrics": "src/phase_a_harness/backend_phase_a_metrics.py",
        "backend_bridge": "src/phase_a_harness/full_synthetic_backend_execution.py",
        "contracts": "src/phase_a_harness/contracts.py",
        "execution_chain_audit": "src/phase_a_harness/phase_a_execution_chain_audit.py",
        "execution_chain_fixture": "src/phase_a_harness/phase_a_execution_chain_fixture.py",
        "independent_verifier": "src/phase_a_harness/full_synthetic_independent_verifier.py",
        "local_metric_models": "src/phase_a_harness/local_metric_models.py",
        "metrics": "src/phase_a_harness/metrics.py",
        "open3d_backend": "src/phase_a_harness/open3d_backend.py",
        "package_init": "src/phase_a_harness/__init__.py",
        "pcl_backend": "src/phase_a_harness/pcl_backend.py",
        "phase_a_attempt_events": "src/phase_a_harness/phase_a_attempt_events.py",
        "phase_a_trial_result_resume": "src/phase_a_harness/phase_a_trial_resume.py",
        "phase_a_trial_result_schema": "src/phase_a_harness/phase_a_trial_result_schema.py",
        "phase_a_trial_result_writer": "src/phase_a_harness/phase_a_trial_result_writer.py",
        "phase_b_generator": "src/phase_a_harness/phase_b_generator.py",
        "phase_b_snapshot_assets": "src/phase_a_harness/phase_b_snapshot_assets.py",
        "phase_b_trial_result": "src/phase_a_harness/phase_b_trial_result.py",
        "protocol_implementation": "src/phase_a_harness/full_synthetic_development_protocol.py",
        "publisher": "src/phase_a_harness/full_synthetic_publisher.py",
        "rotation_metrics": "src/phase_a_harness/rotation_metrics.py",
        "runner": "src/phase_a_harness/full_synthetic_development_runner.py",
        "snapshot_reader": "src/phase_a_harness/snapshot_reader.py",
        "snapshot_builder": "src/phase_a_harness/full_synthetic_snapshot_builder.py",
        "statistics": "src/phase_a_harness/full_synthetic_statistics.py",
        "systematic_offset": "src/phase_a_harness/systematic_offset_analysis.py",
        "trial_bridge": "src/phase_a_harness/full_synthetic_trial_result.py",
        "types": "src/phase_a_harness/types.py",
    }
    script_paths = {
        "analyze_script": "scripts/analyze_full_synthetic_development.py",
        "build_snapshot_script": "scripts/build_full_synthetic_development_snapshots.py",
        "publish_script": "scripts/publish_full_synthetic_development.py",
        "run_script": "scripts/run_full_synthetic_development.py",
        "verify_script": "scripts/verify_full_synthetic_development.py",
    }
    test_paths = {
        f"existing_{index:02d}": path
        for index, path in enumerate(EXISTING_TEST_FILES)
    } | {
        f"new_{index:02d}": path for index, path in enumerate(NEW_TEST_FILES)
    }
    frozen_generator_paths = {
        f"export_{index:02d}": path
        for index, path in enumerate(FROZEN_GENERATOR_RUNTIME_FILES)
    }
    payload: dict[str, Any] = {
        "confirmatory_run_authorized": False,
        "formal_execution_authorized": False,
        "measurement_paper_mainline_authorized": False,
        "real_data_run_authorized": False,
        "backend_parameter_contract_path": "frozen_assets/backend_parameter_contract.json",
        "backend_parameter_contract_sha256": _sha(
            repository, "frozen_assets/backend_parameter_contract.json"
        ),
        "baseline_commit": BASELINE_COMMIT,
        "code_bindings": {
            name: {"path": path, "sha256": _sha(repository, path)}
            for name, path in sorted(code_paths.items())
        },
        "combined_planned_snapshot_count": 1260,
        "combined_planned_snapshots_path": COMBINED_SNAPSHOTS_RELATIVE.as_posix(),
        "combined_planned_snapshots_sha256": _sha(
            repository, COMBINED_SNAPSHOTS_RELATIVE.as_posix()
        ),
        "combined_planned_trial_count": 2520,
        "combined_planned_trials_path": COMBINED_TRIALS_RELATIVE.as_posix(),
        "combined_planned_trials_sha256": _sha(
            repository, COMBINED_TRIALS_RELATIVE.as_posix()
        ),
        "environment_manifest_path": "frozen_assets/environment_manifest.json",
        "environment_manifest_sha256": _sha(
            repository, "frozen_assets/environment_manifest.json"
        ),
        "development_runtime_versions": dict(DEVELOPMENT_RUNTIME_VERSIONS),
        "formal_execution_command": FORMAL_EXECUTION_COMMAND,
        "formal_execution_command_sha256": canonical_json_sha256(
            {"command": FORMAL_EXECUTION_COMMAND}
        ),
        "formal_mamba_root_prefix": FROZEN_MAMBA_ROOT_PREFIX,
        "formal_micromamba_executable": MICROMAMBA_EXECUTABLE,
        "formal_output_dir": protocol["formal_execution"]["output_directory"],
        "formal_pre_run_branch": DEVELOPMENT_BRANCH,
        "formal_pre_run_tag": PRE_RUN_TAG,
        "formal_run_id": protocol["formal_execution"]["run_id"],
        "formal_workers": protocol["formal_execution"]["workers"],
        "generator_export_manifest_path": "frozen_assets/phase_b_generator_export_manifest.csv",
        "generator_export_manifest_sha256": _sha(
            repository, "frozen_assets/phase_b_generator_export_manifest.csv"
        ),
        "frozen_generator_runtime_bindings": {
            name: {"path": path, "sha256": _sha(repository, path)}
            for name, path in sorted(frozen_generator_paths.items())
        },
        "manifest_version": "1",
        "new_planned_snapshot_count": 1050,
        "new_planned_snapshots_path": NEW_SNAPSHOTS_RELATIVE.as_posix(),
        "new_planned_snapshots_sha256": _sha(
            repository, NEW_SNAPSHOTS_RELATIVE.as_posix()
        ),
        "new_planned_trial_count": 2100,
        "new_planned_trials_path": NEW_TRIALS_RELATIVE.as_posix(),
        "new_planned_trials_sha256": _sha(repository, NEW_TRIALS_RELATIVE.as_posix()),
        "new_snapshot_cache_root": SNAPSHOT_CACHE_RELATIVE.as_posix(),
        "new_snapshot_lock_path": SNAPSHOT_LOCK_RELATIVE.as_posix(),
        "new_snapshot_lock_sha256": _sha(repository, SNAPSHOT_LOCK_RELATIVE.as_posix()),
        "open3d_parameter_sha256": "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413",
        "pcl_cli_path": "bin/pcl_point_to_plane_cli",
        "pcl_cli_sha256": _sha(repository, "bin/pcl_point_to_plane_cli"),
        "pcl_parameter_sha256": "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd",
        "phase_a_artifact_sha256s_path": "artifacts/formal_phase_a_v1/SHA256SUMS",
        "phase_a_artifact_sha256s_sha256": _sha(
            repository, "artifacts/formal_phase_a_v1/SHA256SUMS"
        ),
        "phase_a_final_decision_path": "artifacts/formal_phase_a_v1/final_decision.json",
        "phase_a_final_decision_sha256": _sha(
            repository, "artifacts/formal_phase_a_v1/final_decision.json"
        ),
        "phase_a_ideal_raw_manifest_path": "results/formal_phase_a_v1/raw_result_manifest.json",
        "phase_a_ideal_raw_manifest_sha256": _sha(
            repository, "results/formal_phase_a_v1/raw_result_manifest.json"
        ),
        "phase_a_results_root": "results/formal_phase_a_v1",
        "phase_a_snapshot_cache_root": "data/frozen_snapshots",
        "phase_b_pass_final_decision_path": "artifacts/phase_b_signal_v1/final_decision.json",
        "phase_b_pass_final_decision_sha256": _sha(
            repository, "artifacts/phase_b_signal_v1/final_decision.json"
        ),
        "phase_b_pass_artifact_sha256s_path": "artifacts/phase_b_signal_v1/SHA256SUMS",
        "phase_b_pass_artifact_sha256s_sha256": _sha(
            repository, "artifacts/phase_b_signal_v1/SHA256SUMS"
        ),
        "phase_b_pass_artifact_verification_path": "artifacts/phase_b_signal_v1/artifact_verification.json",
        "phase_b_pass_artifact_verification_sha256": _sha(
            repository, "artifacts/phase_b_signal_v1/artifact_verification.json"
        ),
        "phase_b_pass_tag": PHASE_B_PASS_TAG,
        "phase_b_pass_tag_commit": BASELINE_COMMIT,
        "phase_b_reference_raw_manifest_path": "results/phase_b_signal_v1/raw_result_manifest.json",
        "phase_b_reference_raw_manifest_sha256": _sha(
            repository, "results/phase_b_signal_v1/raw_result_manifest.json"
        ),
        "phase_b_reference_results_root": "results/phase_b_signal_v1",
        "phase_b_reference_snapshot_cache_root": "data/phase_b_signal_snapshots",
        "phase_b_subset_reproduction_report_path": PHASE_B_SUBSET_REPORT_RELATIVE.as_posix(),
        "phase_b_subset_reproduction_evidence_path": PHASE_B_SUBSET_EVIDENCE_RELATIVE.as_posix(),
        "dry_run_report_path": DRY_RUN_REPORT_RELATIVE.as_posix(),
        "existing_test_results_path": EXISTING_TEST_RESULTS_RELATIVE.as_posix(),
        "new_test_results_path": NEW_TEST_RESULTS_RELATIVE.as_posix(),
        "phase_b_subset_execution_authorized": True,
        "pre_run_gate_report_path": PRE_RUN_GATE_REPORT_RELATIVE.as_posix(),
        "protocol_document_path": "docs/full_synthetic_development_protocol_v1.md",
        "protocol_document_sha256": _sha(
            repository, "docs/full_synthetic_development_protocol_v1.md"
        ),
        "scientific_protocol_path": PROTOCOL_RELATIVE.as_posix(),
        "scientific_protocol_sha256": _sha(repository, PROTOCOL_RELATIVE.as_posix()),
        "script_bindings": {
            name: {"path": path, "sha256": _sha(repository, path)}
            for name, path in sorted(script_paths.items())
        },
        "test_bindings": {
            name: {"path": path, "sha256": _sha(repository, path)}
            for name, path in sorted(test_paths.items())
        },
        "test_execution_contract": {
            "existing": {
                "collection_sha256": EXISTING_TEST_COLLECTION_SHA256,
                "command": EXISTING_TEST_COMMAND,
                "command_sha256": canonical_json_sha256(
                    {"command": EXISTING_TEST_COMMAND}
                ),
                "exact_test_count": 65,
                "files": list(EXISTING_TEST_FILES),
            },
            "new": {
                "collection_sha256": NEW_TEST_COLLECTION_SHA256,
                "command": NEW_TEST_COMMAND,
                "command_sha256": canonical_json_sha256(
                    {"command": NEW_TEST_COMMAND}
                ),
                "exact_test_count": 35,
                "files": list(NEW_TEST_FILES),
                "maximum_test_count": 35,
                "minimum_test_count": 25,
                "required_categories": list(REQUIRED_NEW_TEST_CATEGORIES),
            },
        },
        "trial_schema_path": "frozen_assets/trial_result_schema.json",
        "trial_schema_sha256": _sha(repository, "frozen_assets/trial_result_schema.json"),
    }
    payload["implementation_contract_sha256"] = canonical_json_sha256(
        {
            "backend_parameter_contract_sha256": payload[
                "backend_parameter_contract_sha256"
            ],
            "code_bindings": payload["code_bindings"],
            "frozen_generator_runtime_bindings": payload[
                "frozen_generator_runtime_bindings"
            ],
            "environment_manifest_sha256": payload["environment_manifest_sha256"],
            "development_runtime_versions": payload["development_runtime_versions"],
            "formal_execution_command_sha256": payload[
                "formal_execution_command_sha256"
            ],
            "formal_mamba_root_prefix": payload["formal_mamba_root_prefix"],
            "formal_micromamba_executable": payload[
                "formal_micromamba_executable"
            ],
            "open3d_parameter_sha256": payload["open3d_parameter_sha256"],
            "pcl_cli_sha256": payload["pcl_cli_sha256"],
            "pcl_parameter_sha256": payload["pcl_parameter_sha256"],
            "script_bindings": payload["script_bindings"],
            "scientific_protocol_sha256": payload["scientific_protocol_sha256"],
            "test_bindings": payload["test_bindings"],
            "test_execution_contract": payload["test_execution_contract"],
            "trial_schema_sha256": payload["trial_schema_sha256"],
        }
    )
    return payload


def create_unauthorized_full_synthetic_manifest(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = repository / MANIFEST_RELATIVE
    if destination.exists():
        raise FileExistsError("refusing to replace the one Development manifest")
    payload = build_full_synthetic_manifest_payload(repository)
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    write_json(destination, payload)
    return payload


def _authorized_evidence_bindings(
    repository: Path, gates: Mapping[str, Any]
) -> dict[str, Any]:
    gate_path = repository / PRE_RUN_GATE_REPORT_RELATIVE
    evidence = gates.get("evidence_sha256")
    if type(evidence) is not dict or not evidence:
        raise ValueError("pre-run gate did not expose raw evidence SHA bindings")
    for relative, digest in evidence.items():
        candidate = (repository / str(relative)).resolve()
        if (
            type(relative) is not str
            or type(digest) is not str
            or repository not in candidate.parents
            or file_sha256(candidate) != digest
        ):
            raise ValueError("pre-run raw evidence SHA binding changed")
    return {
        "pre_run_evidence_sha256": dict(sorted(evidence.items())),
        "pre_run_gate_report_payload_sha256": canonical_json_sha256(dict(gates)),
        "pre_run_gate_report_sha256": file_sha256(gate_path),
    }


def expected_full_synthetic_manifest_payload(
    root: str | Path, *, authorized: bool
) -> dict[str, Any]:
    repository = Path(root).resolve()
    expected = build_full_synthetic_manifest_payload(repository)
    if authorized:
        gates = derive_full_synthetic_pre_run_gate_report(repository)
        gate_path = repository / PRE_RUN_GATE_REPORT_RELATIVE
        recorded = _load_json_object_strict(gate_path, "pre-run gate report")
        if recorded != gates:
            raise ValueError("recorded pre-run gate report differs from raw evidence")
        if any(gates.get(name) is not True for name in REQUIRED_PRE_RUN_GATES):
            raise PermissionError("authorized manifest raw gates no longer pass")
        expected["formal_execution_authorized"] = True
        expected.update(_authorized_evidence_bindings(repository, gates))
    expected["manifest_payload_sha256"] = canonical_json_sha256(expected)
    return expected


def load_strict_authorized_full_synthetic_manifest(
    manifest_path: str | Path,
    *,
    require_operational_git_gate: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Load the sole authorized manifest and rederive every bound dependency.

    Analysis, independent verification, and publication share this entry so a
    payload-SHA-valid but stale or partially forged manifest is never accepted.
    The optional operational switch exists only for isolated unit tests; formal
    entry points retain the default and therefore require tag==HEAD and clean.
    """

    candidate = Path(manifest_path).resolve()
    if candidate.name != MANIFEST_RELATIVE.name:
        raise ValueError("Full Synthetic must use its one v1 manifest")
    repository = manifest_root(candidate)
    if candidate != (repository / MANIFEST_RELATIVE).resolve():
        raise ValueError("Full Synthetic manifest path is not canonical")
    manifests = sorted(
        path.resolve()
        for path in candidate.parent.glob("full_synthetic_development_manifest*.json")
        if path.is_file()
    )
    if manifests != [candidate]:
        raise ValueError("multiple or ambiguous Development manifests exist")
    actual = _load_json_object_strict(candidate, "Development manifest")
    if actual.get("manifest_version") != "1":
        raise ValueError("frozen experiment manifest identity mismatch")
    if actual.get("formal_execution_authorized") is not True:
        raise PermissionError("formal execution is not authorized")
    stored = actual.get("manifest_payload_sha256")
    unsigned = {
        key: value for key, value in actual.items() if key != "manifest_payload_sha256"
    }
    if stored != canonical_json_sha256(unsigned):
        raise ValueError("frozen experiment manifest payload SHA mismatch")
    expected = expected_full_synthetic_manifest_payload(repository, authorized=True)
    if actual != expected:
        raise ValueError("Development manifest differs from rederived authorized payload")
    assert_isolated_python_runtime()
    verify_frozen_runtime_environment(repository)
    if require_operational_git_gate:
        verify_formal_operational_git_gate(repository)
    return candidate, actual


def _junit_collection_sha256(testcases: Sequence[str]) -> str:
    return canonical_json_sha256({"testcases": sorted(testcases)})


def verify_junit_contract(
    path: Path,
    *,
    expected_count: int,
    expected_collection_sha256: str,
) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return {
            "pass": False,
            "collection_sha256": None,
            "duplicate_testcase_count": 0,
            "error_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "test_count": 0,
        }
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    testcases = list(root.iter("testcase"))
    identities = [
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
        for case in testcases
    ]
    failures = sum(case.find("failure") is not None for case in testcases)
    errors = sum(case.find("error") is not None for case in testcases)
    skipped = sum(case.find("skipped") is not None for case in testcases)
    declared_tests = sum(int(suite.attrib.get("tests", "-1")) for suite in suites)
    declared_failures = sum(int(suite.attrib.get("failures", "-1")) for suite in suites)
    declared_errors = sum(int(suite.attrib.get("errors", "-1")) for suite in suites)
    declared_skipped = sum(int(suite.attrib.get("skipped", "-1")) for suite in suites)
    duplicate_count = len(identities) - len(set(identities))
    collection_sha = _junit_collection_sha256(identities)
    valid_expected_sha = bool(
        len(expected_collection_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_collection_sha256)
    )
    passed = bool(
        valid_expected_sha
        and len(testcases) == expected_count
        and declared_tests == expected_count
        and failures == errors == skipped == duplicate_count == 0
        and declared_failures == declared_errors == declared_skipped == 0
        and all(case.attrib.get("classname") and case.attrib.get("name") for case in testcases)
        and collection_sha == expected_collection_sha256
    )
    return {
        "pass": passed,
        "collection_sha256": collection_sha,
        "declared_test_count": declared_tests,
        "duplicate_testcase_count": duplicate_count,
        "error_count": errors,
        "failure_count": failures,
        "skipped_count": skipped,
        "test_count": len(testcases),
    }


def _derive_full_synthetic_pre_execution_evidence(
    root: str | Path, *, include_subset_reproduction: bool
) -> dict[str, Any]:
    repository = Path(root).resolve()
    protocol = verify_full_synthetic_protocol_file(repository)
    phase_a = verify_phase_a_ideal_import(repository, write_report=False)
    recorded_phase_a = _load_json_object_strict(
        repository / PHASE_A_IMPORT_REPORT_RELATIVE,
        "recorded Phase A IDEAL import report",
    )
    phase_a_record_match = recorded_phase_a == phase_a
    phase_b_archive = verify_phase_b_pass_archive(repository)
    preparation_path = repository / PREPARATION_REPORT_RELATIVE
    dry_path = repository / DRY_RUN_REPORT_RELATIVE
    preparation = _load_json_object_strict(
        preparation_path, "Full Synthetic snapshot preparation report"
    )
    dry = _load_json_object_strict(dry_path, "Full Synthetic dry-run report")
    existing = verify_junit_contract(
        repository / EXISTING_TEST_RESULTS_RELATIVE,
        expected_count=65,
        expected_collection_sha256=EXISTING_TEST_COLLECTION_SHA256,
    )
    new = verify_junit_contract(
        repository / NEW_TEST_RESULTS_RELATIVE,
        expected_count=35,
        expected_collection_sha256=NEW_TEST_COLLECTION_SHA256,
    )
    source_isolation = all(
        int(report.get(field, -1)) == 0
        for report in (preparation, dry)
        for field in (
            "source_repository_runtime_file_read_count",
            "source_repository_runtime_import_count",
        )
    )
    zero_execution = all(
        int(dry.get(field, -1)) == 0
        for field in (
            "backend_execution_count",
            "formal_rng_access_count",
            "started_event_count",
            "trial_result_count",
        )
    )
    preparation_zero_fields = (
        "confirmatory_seed_instantiation_count",
        "generator_reproduction_checksum_mismatch_count",
        "generator_reproduction_confirmatory_seed_instantiation_count",
        "generator_reproduction_gt_optimization_leakage_count",
        "generator_reproduction_old_capture_range_seed_access_count",
        "snapshot_checksum_mismatch_count",
        "snapshot_corrupt_count",
        "snapshot_duplicate_count",
        "snapshot_extra_count",
        "snapshot_file_sha_mismatch_count",
        "snapshot_missing_count",
    )
    preparation_direct = bool(
        preparation.get("schema_version") == "full_synthetic_snapshot_preparation_v1"
        and preparation.get("new_snapshot_count") == 1050
        and preparation.get("new_trial_count") == 2100
        and preparation.get("NEW_1050_SNAPSHOTS_COMPLETE") is True
        and preparation.get("NEW_2100_TRIALS_PLANNED") is True
        and preparation.get("COMBINED_1260_SNAPSHOTS_PLANNED") is True
        and preparation.get("COMBINED_2520_TRIALS_PLANNED") is True
        and preparation.get("GENERATOR_REGRESSION_PASS") is True
        and preparation.get("PHASE_A_IDEAL_IMPORT_PASS") is True
        and preparation.get("PHASE_B_SNAPSHOT_SUBSET_REPRODUCTION_PASS") is True
        and preparation.get("phase_b_overlap_snapshot_count") == 42
        and preparation.get("scientific_protocol_sha256")
        == protocol["protocol_sha256"]
        and all(int(preparation.get(name, -1)) == 0 for name in preparation_zero_fields)
    )
    expected_condition_counts = {condition: 420 for condition in NEW_CONDITIONS}
    dry_direct = bool(
        dry.get("schema_version") == "full_synthetic_development_dry_run_v1"
        and dry.get("run_id") == "full-synthetic-development-v1"
        and dry.get("workers") == 2
        and Path(str(dry.get("output_dir", ""))).resolve()
        == (repository / "results/full_synthetic_development_v1").resolve()
        and dry.get("new_snapshot_count") == 1050
        and dry.get("new_trial_count") == 2100
        and dry.get("combined_snapshot_count") == 1260
        and dry.get("combined_trial_count") == 2520
        and dry.get("open3d_trial_count") == 1050
        and dry.get("pcl_trial_count") == 1050
        and dry.get("native_trial_count") == 0
        and dry.get("condition_trial_counts") == expected_condition_counts
        and dry.get("phase_b_subset_snapshot_count") == 42
        and dry.get("phase_b_subset_trial_count") == 84
        and dry.get("snapshot_backend_pairing_mismatch_count") == 0
        and dry.get("PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS") is True
        and dry.get("FULL_SYNTHETIC_DRY_RUN_PASS") is True
    )
    subset_path = repository / PHASE_B_SUBSET_REPORT_RELATIVE
    subset: dict[str, Any] = {}
    if include_subset_reproduction:
        from .full_synthetic_development_runner import (
            verify_phase_b_trial_subset_reproduction,
        )

        subset = verify_phase_b_trial_subset_reproduction(
            repository, repository / "results/full_synthetic_development_v1"
        )
    subset_direct = bool(
        include_subset_reproduction
        and subset.get("schema_version")
        == "full_synthetic_phase_b_subset_reproduction_v1"
        and subset.get("PHASE_B_SUBSET_REPRODUCTION_PASS") is True
        and subset.get("compared_trial_count") == 84
        and subset.get("missing_trial_count") == 0
        and subset.get("extra_trial_count") == 0
        and subset.get("mismatch_count") == 0
        and subset.get("immutable_subset_evidence_pass") is True
        and type(subset.get("immutable_subset_evidence_sha256")) is str
        and subset.get("absolute_tolerance") == 1e-12
        and subset.get("relative_tolerance") == 1e-12
    )
    evidence_paths = [
        preparation_path,
        dry_path,
        repository / EXISTING_TEST_RESULTS_RELATIVE,
        repository / NEW_TEST_RESULTS_RELATIVE,
        repository / PROTOCOL_RELATIVE,
        repository / PHASE_A_IMPORT_REPORT_RELATIVE,
        repository / "artifacts/formal_phase_a_v1/SHA256SUMS",
        repository / "artifacts/formal_phase_a_v1/artifact_verification.json",
        repository / "artifacts/formal_phase_a_v1/final_decision.json",
        repository / "results/formal_phase_a_v1/raw_result_manifest.json",
        repository / "results/formal_phase_a_v1/run_manifest.json",
        repository / "artifacts/phase_b_signal_v1/SHA256SUMS",
        repository / "artifacts/phase_b_signal_v1/artifact_verification.json",
        repository / "artifacts/phase_b_signal_v1/final_decision.json",
        repository / "results/phase_b_signal_v1/raw_result_manifest.json",
    ]
    if include_subset_reproduction:
        evidence_paths.extend(
            [
                subset_path,
                repository / PHASE_B_SUBSET_EVIDENCE_RELATIVE,
            ]
        )
    report = {
        "DRY_RUN_ZERO_EXECUTION_PASS": zero_execution,
        "FULL_SYNTHETIC_DRY_RUN_PASS": dry_direct,
        "FULL_SYNTHETIC_EXISTING_TEST_PASS": existing["pass"],
        "FULL_SYNTHETIC_NEW_TEST_PASS": new["pass"],
        "FULL_SYNTHETIC_PLAN_PASS": preparation_direct,
        "FULL_SYNTHETIC_TEST_PASS": existing["pass"] and new["pass"],
        "GENERATOR_REGRESSION_PASS": preparation_direct,
        "NEW_1050_SNAPSHOTS_COMPLETE": preparation_direct,
        "PHASE_A_IDEAL_IMPORT_PASS": bool(
            phase_a.get("PHASE_A_IDEAL_IMPORT_PASS") is True
            and phase_a_record_match
        ),
        "PHASE_B_PASS_ARCHIVE_VALID": phase_b_archive["PHASE_B_PASS_ARCHIVE_VALID"],
        "PHASE_B_SUBSET_REPRODUCTION_DESIGN_PASS": dry_direct,
        "PHASE_B_SUBSET_REPRODUCTION_PASS": subset_direct,
        "SCIENTIFIC_PROTOCOL_FROZEN_PASS": protocol[
            "SCIENTIFIC_PROTOCOL_FROZEN_PASS"
        ],
        "SOURCE_RUNTIME_ISOLATION_PASS": source_isolation,
        "evidence_sha256": {
            path.relative_to(repository).as_posix(): file_sha256(path)
            for path in evidence_paths
        },
        "existing_test_collection_sha256": existing["collection_sha256"],
        "existing_test_command": EXISTING_TEST_COMMAND,
        "existing_test_command_sha256": canonical_json_sha256(
            {"command": EXISTING_TEST_COMMAND}
        ),
        "existing_test_count": existing["test_count"],
        "formal_execution_authorized_before_transition": False,
        "new_test_collection_sha256": new["collection_sha256"],
        "new_test_command": NEW_TEST_COMMAND,
        "new_test_command_sha256": canonical_json_sha256(
            {"command": NEW_TEST_COMMAND}
        ),
        "new_test_count": new["test_count"],
        "phase_a_import_audit": phase_a,
        "phase_b_archive_audit": phase_b_archive,
        "schema_version": "full_synthetic_development_pre_run_gate_v1",
    }
    return report


def derive_full_synthetic_pre_subset_gate_report(
    root: str | Path,
) -> dict[str, Any]:
    report = _derive_full_synthetic_pre_execution_evidence(
        root, include_subset_reproduction=False
    )
    report["ALL_PRE_SUBSET_GATES_PASS"] = all(
        report.get(name) is True for name in REQUIRED_PRE_SUBSET_GATES
    )
    return report


def derive_full_synthetic_pre_run_gate_report(root: str | Path) -> dict[str, Any]:
    """Recompute every pre-run gate from bound raw evidence."""

    report = _derive_full_synthetic_pre_execution_evidence(
        root, include_subset_reproduction=True
    )
    report["ALL_PRE_RUN_GATES_PASS"] = all(
        report.get(name) is True for name in REQUIRED_PRE_RUN_GATES
    )
    return report


def authorize_full_synthetic_manifest_once(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    destination = repository / MANIFEST_RELATIVE
    _, current = load_manifest(destination, require_authorized=False)
    if current.get("formal_execution_authorized") is not False:
        raise PermissionError("Development authorization is not a one-way false-to-true transition")
    expected = expected_full_synthetic_manifest_payload(repository, authorized=False)
    if current != expected:
        raise ValueError("Development manifest changed before authorization")
    gates = derive_full_synthetic_pre_run_gate_report(repository)
    gate_path = repository / PRE_RUN_GATE_REPORT_RELATIVE
    if gate_path.exists():
        recorded = _load_json_object_strict(gate_path, "pre-run gate report")
        if recorded != gates:
            raise ValueError("recorded pre-run gate report is not derived from raw evidence")
    else:
        write_json(gate_path, gates)
    if any(gates.get(name) is not True for name in REQUIRED_PRE_RUN_GATES):
        raise PermissionError("Development pre-run gates are incomplete")
    authorized = dict(current)
    authorized.pop("manifest_payload_sha256")
    authorized["formal_execution_authorized"] = True
    authorized.update(_authorized_evidence_bindings(repository, gates))
    authorized["manifest_payload_sha256"] = canonical_json_sha256(authorized)
    write_json(destination, authorized)
    return authorized


__all__ = [
    "ALL_CONDITIONS",
    "BACKENDS",
    "COMBINED_SNAPSHOTS_RELATIVE",
    "COMBINED_TRIALS_RELATIVE",
    "CONDITION_PARAMETERS",
    "DEVELOPMENT_BRANCH",
    "DEVELOPMENT_RUNTIME_VERSIONS",
    "DRY_RUN_REPORT_RELATIVE",
    "EXISTING_TEST_COLLECTION_SHA256",
    "EXISTING_TEST_COMMAND",
    "EXISTING_TEST_FILES",
    "FORMAL_EXECUTION_COMMAND",
    "FROZEN_MAMBA_ROOT_PREFIX",
    "GEOMETRY_SEEDS",
    "MANIFEST_RELATIVE",
    "MEASUREMENT_SEEDS",
    "NEW_CONDITIONS",
    "NEW_SNAPSHOTS_RELATIVE",
    "NEW_TRIALS_RELATIVE",
    "NEW_TEST_COLLECTION_SHA256",
    "NEW_TEST_COMMAND",
    "NEW_TEST_FILES",
    "PHASE_B_PASS_TAG",
    "PHASE_B_OVERLAP_CONDITIONS",
    "PHASE_B_SUBSET_EVIDENCE_RELATIVE",
    "PHASE_B_SUBSET_REPORT_RELATIVE",
    "PRE_RUN_GATE_REPORT_RELATIVE",
    "PRE_RUN_TAG",
    "PROTOCOL_RELATIVE",
    "REQUIRED_NEW_TEST_CATEGORIES",
    "REPEAT_INDICES",
    "SCENES",
    "SNAPSHOT_CACHE_RELATIVE",
    "SNAPSHOT_COLUMNS",
    "SNAPSHOT_LOCK_RELATIVE",
    "TRIAL_COLUMNS",
    "authorize_full_synthetic_manifest_once",
    "assert_isolated_python_runtime",
    "build_full_synthetic_manifest_payload",
    "create_unauthorized_full_synthetic_manifest",
    "derive_full_synthetic_pre_run_gate_report",
    "derive_full_synthetic_pre_subset_gate_report",
    "expected_full_synthetic_manifest_payload",
    "full_synthetic_protocol_payload",
    "is_phase_b_overlap_key",
    "load_strict_authorized_full_synthetic_manifest",
    "phase_b_overlap_snapshots",
    "planned_combined_snapshots",
    "planned_new_snapshots",
    "planned_trials",
    "read_full_synthetic_plans",
    "snapshot_id_for",
    "validate_full_synthetic_plans",
    "verify_phase_a_ideal_import",
    "verify_formal_operational_git_gate",
    "verify_frozen_runtime_environment",
    "verify_full_synthetic_protocol_file",
    "verify_junit_contract",
    "verify_phase_b_pass_archive",
    "write_full_synthetic_plans",
    "write_full_synthetic_protocol",
]
