"""Bounded, Development-only qualification for Synthetic Confirmatory v2.

This module deliberately separates qualification evidence from formal v2
science.  It can construct only Development-seed snapshots, execute only
qualification-labelled backend controls, and optionally run the seed-free
3-snapshot/6-trial fixture chain.  It never calls ``build_v2_snapshot``, never
opens the formal v2 snapshot/result directories, and never writes a formal
trial result.

The expensive 1,050-snapshot regression is explicit.  It compares the exact
scientific payload (arrays, checksums, identity, frozen noise/dropout
parameters, and realized record checksums where the historical cache exposes
them) while excluding file-layout and metadata-envelope fields from the
scientific comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_json_sha256, file_sha256
from .synthetic_confirmatory_v2_contract import (
    BOOTSTRAP_SEED,
    GEOMETRY_SEEDS as FORMAL_V2_GEOMETRY_SEEDS,
    MEASUREMENT_SEEDS as FORMAL_V2_MEASUREMENT_SEEDS,
    OLD_V1_BOOTSTRAP_SEED,
    OLD_V1_GEOMETRY_SEEDS,
    OLD_V1_MEASUREMENT_SEEDS,
    SCENES,
)


QUALIFICATION_SCHEMA = "synthetic_confirmatory_v2_qualification_report_v1"
QUALIFICATION_PLAN_SCHEMA = "synthetic_confirmatory_v2_qualification_plan_v1"
QUALIFICATION_LOCK_SCHEMA = "synthetic_confirmatory_v2_qualification_lock_v1"
DEVELOPMENT_REGRESSION_SCHEMA = (
    "synthetic_confirmatory_v2_development_nonideal_regression_v1"
)

IDEAL_QUALIFICATION_SNAPSHOT_COUNT = 21
IDEAL_QUALIFICATION_TRIAL_COUNT = 42
INDEPENDENT_NEGATIVE_SNAPSHOT_COUNT = 21
DEVELOPMENT_NONIDEAL_SNAPSHOT_COUNT = 1050
TRANSLATION_THRESHOLD_M = 0.001
ROTATION_THRESHOLD_RAD = 0.00017453292519943296
QUANTILE_METHOD = "linear"

EXPECTED_BACKEND_PARAMETER_CONTRACT_SHA256 = (
    "6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9"
)
EXPECTED_OPEN3D_PARAMETER_SHA256 = (
    "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
)
EXPECTED_PCL_PARAMETER_SHA256 = (
    "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
)
EXPECTED_PCL_CLI_SHA256 = (
    "d42ce655df74117f0e6965c9df1326526fabba9f4e65ed10a5644ed911fad7ff"
)

FORMAL_ZERO_COUNTERS = (
    "FORMAL_V2_SEED_RNG_ACCESS_COUNT",
    "FORMAL_V2_SNAPSHOT_ACCESS_COUNT",
    "FORMAL_V2_SNAPSHOT_GENERATION_COUNT",
    "FORMAL_V2_BACKEND_EXECUTION_COUNT",
    "FORMAL_V2_TRIAL_RESULT_COUNT",
    "NATIVE_EXECUTION_COUNT",
)

_FORMAL_AND_RETIRED_SEEDS = frozenset(
    (
        *FORMAL_V2_GEOMETRY_SEEDS,
        *FORMAL_V2_MEASUREMENT_SEEDS,
        BOOTSTRAP_SEED,
        *OLD_V1_GEOMETRY_SEEDS,
        *OLD_V1_MEASUREMENT_SEEDS,
        OLD_V1_BOOTSTRAP_SEED,
    )
)

_SCIENTIFIC_METADATA_FIELDS = (
    "condition",
    "development_protocol_sha256",
    "dropout_parameters",
    "generator_sha256",
    "geometry_seed",
    "geometry_seed_index",
    "independent_sampling",
    "initial_pose",
    "measurement_seed",
    "measurement_seed_index",
    "noise_parameters",
    "reference_pose_checksum",
    "repeat_index",
    "scene_variant",
    "snapshot_builder_sha256",
    "snapshot_id",
    "source_checksum",
    "source_is_target_subset",
    "source_point_count",
    "target_checksum",
    "target_point_count",
)

_METADATA_ENVELOPE_FIELDS = (
    "array_file_sha256",
    "metadata_payload_sha256",
    "schema_version",
    "snapshot_schema_version",
    "snapshot_builder_contract_version",
    "planned_snapshot_id",
    "snapshot_checksum",
)


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _raw_sha256(value: Any) -> str:
    import numpy as np

    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("qualification checksum input must be C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _zero_formal_counters() -> dict[str, int]:
    return {name: 0 for name in FORMAL_ZERO_COUNTERS}


def _assert_zero_formal_counters(counters: Mapping[str, Any]) -> None:
    if set(counters) != set(FORMAL_ZERO_COUNTERS) or any(
        type(counters[name]) is not int or counters[name] != 0
        for name in FORMAL_ZERO_COUNTERS
    ):
        raise PermissionError("formal v2 execution counter changed during qualification")


def _assert_development_seed(value: int, allowed: Sequence[int], label: str) -> int:
    candidate = int(value)
    if candidate in _FORMAL_AND_RETIRED_SEEDS:
        raise PermissionError(f"Confirmatory seed reached Development qualification: {label}")
    if candidate not in {int(item) for item in allowed}:
        raise PermissionError(f"non-Development seed reached qualification: {label}")
    return candidate


def _inside(candidate: Path, parent: Path) -> bool:
    resolved = candidate.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


class QualificationAccessMonitor:
    """Reject source-repository and formal-v2 data/result reads."""

    def __init__(self, repository: str | Path) -> None:
        from .contracts import SOURCE_REPOSITORY

        root = Path(repository).resolve()
        self._source = SOURCE_REPOSITORY.resolve()
        self._formal_snapshot = (root / "data/synthetic_confirmatory_v2_snapshots").resolve()
        self._formal_result = (root / "results/synthetic_confirmatory_v2").resolve()
        self._lock = threading.Lock()
        self.source_repository_runtime_file_read_count = 0
        self.formal_v2_snapshot_file_read_count = 0
        self.formal_v2_result_file_read_count = 0
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or not args or not isinstance(args[0], (str, bytes)):
                return
            try:
                candidate = Path(args[0]).resolve()
            except (OSError, TypeError):
                return
            label = None
            if _inside(candidate, self._source):
                label = "source"
            elif _inside(candidate, self._formal_snapshot):
                label = "formal_snapshot"
            elif _inside(candidate, self._formal_result):
                label = "formal_result"
            if label is None:
                return
            with self._lock:
                if label == "source":
                    self.source_repository_runtime_file_read_count += 1
                elif label == "formal_snapshot":
                    self.formal_v2_snapshot_file_read_count += 1
                else:
                    self.formal_v2_result_file_read_count += 1
            raise PermissionError(f"qualification attempted forbidden {label} read: {candidate}")

        sys.addaudithook(audit)
        self._installed = True

    def report(self) -> dict[str, int]:
        return {
            "formal_v2_result_file_read_count": self.formal_v2_result_file_read_count,
            "formal_v2_snapshot_file_read_count": self.formal_v2_snapshot_file_read_count,
            "source_repository_runtime_file_read_count": (
                self.source_repository_runtime_file_read_count
            ),
        }


def qualification_execution_binding(root: str | Path) -> dict[str, Any]:
    """Bind the exact qualification implementation without authorizing formal v2."""

    repository = Path(root).resolve()
    relative_paths = {
        "development_protocol": "configs/zero_perturbation/development_v1.yaml",
        "frozen_scene_generator": (
            "src/phase_a_harness/phase_b_generator_frozen/capture_range/"
            "day2_development_scene.py"
        ),
        "frozen_snapshot_builder": (
            "src/phase_a_harness/phase_b_generator_frozen/zero_perturbation/"
            "snapshot_builder.py"
        ),
        "v2_snapshot_builder": (
            "src/phase_a_harness/synthetic_confirmatory_v2_snapshot_builder.py"
        ),
        "v2_contract": "src/phase_a_harness/synthetic_confirmatory_v2_contract.py",
        "qualification_orchestrator": (
            "src/phase_a_harness/synthetic_confirmatory_v2_qualification.py"
        ),
        "qualification_script": "scripts/qualify_synthetic_confirmatory_v2.py",
        "backend_parameter_contract": "frozen_assets/backend_parameter_contract.json",
        "open3d_adapter": "src/phase_a_harness/open3d_backend.py",
        "pcl_adapter": "src/phase_a_harness/pcl_backend.py",
        "backend_execution": "src/phase_a_harness/phase_a_execution_chain_audit.py",
        "trial_schema_validator": "src/phase_a_harness/phase_a_trial_result_schema.py",
        "pcl_cli": "bin/pcl_point_to_plane_cli",
        "fixture_plan": "frozen_assets/fixtures/fixture_plan.json",
        "fixture_snapshot_lock": "frozen_assets/fixtures/fixture_snapshot_lock.json",
        "fixture_parameter_lock": (
            "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
        ),
        "fixture_qualification": "src/phase_a_harness/fixture_qualification.py",
    }
    files: dict[str, dict[str, str]] = {}
    for name, relative in relative_paths.items():
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"qualification binding input missing: {relative}")
        files[name] = {"path": relative, "sha256": file_sha256(path)}
    if (
        files["backend_parameter_contract"]["sha256"]
        != EXPECTED_BACKEND_PARAMETER_CONTRACT_SHA256
        or files["pcl_cli"]["sha256"] != EXPECTED_PCL_CLI_SHA256
    ):
        raise ValueError("frozen backend qualification input SHA changed")
    core = {
        "files": files,
        "formal_v2_execution_authorized": False,
        "qualification_only": True,
        "schema_version": "synthetic_confirmatory_v2_qualification_binding_v1",
    }
    return {**core, "qualification_binding_sha256": canonical_json_sha256(core)}


def qualification_execution_plan(root: str | Path) -> dict[str, Any]:
    """Return an RNG-free/backend-free plan and expected runtime envelope."""

    from .synthetic_confirmatory_v2_snapshot_builder import (
        DEVELOPMENT_GEOMETRY_SEEDS,
    )

    development = tuple(
        _assert_development_seed(seed, DEVELOPMENT_GEOMETRY_SEEDS, "geometry")
        for seed in DEVELOPMENT_GEOMETRY_SEEDS
    )
    formal_counters = _zero_formal_counters()
    _assert_zero_formal_counters(formal_counters)
    return {
        "DEVELOPMENT_SEED_DISJOINT_FROM_CONFIRMATORY_PASS": not bool(
            set(development) & _FORMAL_AND_RETIRED_SEEDS
        ),
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        "formal_v2_access_counters": formal_counters,
        "ideal_backend_control_trial_count": IDEAL_QUALIFICATION_TRIAL_COUNT,
        "ideal_geometry_snapshot_count": IDEAL_QUALIFICATION_SNAPSHOT_COUNT,
        "independent_negative_snapshot_count": INDEPENDENT_NEGATIVE_SNAPSHOT_COUNT,
        "native_trial_count": 0,
        "nonideal_development_regression_snapshot_count": (
            DEVELOPMENT_NONIDEAL_SNAPSHOT_COUNT
        ),
        "qualification_binding": qualification_execution_binding(root),
        "qualification_only": True,
        "runtime_expectation": {
            "fixture_chain": "less than 1 minute",
            "ideal_and_independent_snapshot_construction": "typically less than 1 minute",
            "ideal_dual_backend_controls": "typically less than 1 minute with two workers",
            "peak_workers": 2,
            "regression_1050_snapshot_rebuild": (
                "approximately 25-50 minutes with two workers because both the v2 "
                "candidate and independent Development record oracle are rebuilt"
            ),
            "total": "approximately 30-55 minutes; hardware and filesystem dependent",
        },
        "schema_version": QUALIFICATION_PLAN_SCHEMA,
    }


def _qualification_snapshot_lock(
    snapshots: Sequence[Mapping[str, Any]], *, condition: str, lineage_expected: bool
) -> dict[str, Any]:
    entries = []
    for value in snapshots:
        metadata = value["metadata"]
        parent = value.get("parent_indices")
        entries.append(
            {
                "condition": metadata["condition"],
                "firewall_audit": dict(value["firewall_audit"]),
                "geometry_seed": metadata["geometry_seed"],
                "measurement_seed": metadata["measurement_seed"],
                "parent_index_count": 0 if parent is None else int(len(parent)),
                "parent_index_sha256": None if parent is None else _raw_sha256(parent),
                "reference_pose_checksum": metadata["reference_pose_checksum"],
                "repeat_index": metadata["repeat_index"],
                "scene_variant": metadata["scene_variant"],
                "snapshot_checksum": metadata["snapshot_checksum"],
                "snapshot_id": metadata["snapshot_id"],
                "source_checksum": metadata["source_checksum"],
                "source_has_target_parent_lineage": metadata[
                    "source_has_target_parent_lineage"
                ],
                "target_checksum": metadata["target_checksum"],
            }
        )
    core = {
        "condition": condition,
        "entries": entries,
        "formal_v2_snapshot_lock": False,
        "lineage_expected": lineage_expected,
        "planned_snapshot_count": len(entries),
        "qualification_only": True,
        "schema_version": QUALIFICATION_LOCK_SCHEMA,
    }
    return {**core, "qualification_lock_sha256": canonical_json_sha256(core)}


def build_ideal_geometry_qualification_snapshots(
    root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 7 x 3 Development-seed, geometry-only IDEAL controls."""

    import numpy as np

    from .synthetic_confirmatory_v2_snapshot_builder import (
        DEVELOPMENT_GEOMETRY_SEEDS,
        build_development_ideal_qualification_snapshot,
    )

    snapshots: list[dict[str, Any]] = []
    violations: list[str] = []
    for scene in SCENES:
        for seed in DEVELOPMENT_GEOMETRY_SEEDS:
            geometry_seed = _assert_development_seed(
                seed, DEVELOPMENT_GEOMETRY_SEEDS, "IDEAL geometry"
            )
            value = build_development_ideal_qualification_snapshot(
                root, scene=scene, geometry_seed=geometry_seed
            )
            metadata = value["metadata"]
            firewall = value["firewall_audit"]
            parent = value["parent_indices"]
            valid = bool(
                metadata["condition"] == "IDEAL_MATCHED"
                and metadata["measurement_seed"] is None
                and metadata["repeat_index"] == 0
                and metadata["source_has_target_parent_lineage"] is True
                and metadata["source_is_target_subset"] is True
                and metadata["lineage_closure_violation_count"] == 0
                and metadata["quantization_closure_pass"] is True
                and parent is not None
                and len(parent) == len(value["source"])
                and len(np.unique(parent)) == len(parent)
                and firewall["measurement_seed_access_count"] == 0
                and firewall["repeat_randomness_count"] == 0
                and firewall["rng_instantiation_count"] == 0
            )
            if not valid:
                violations.append(str(metadata.get("snapshot_id")))
            snapshots.append(value)
    lock = _qualification_snapshot_lock(
        snapshots, condition="IDEAL_MATCHED", lineage_expected=True
    )
    report = {
        "IDEAL_GEOMETRY_QUALIFICATION_PASS": bool(
            len(snapshots) == IDEAL_QUALIFICATION_SNAPSHOT_COUNT
            and len({row["metadata"]["snapshot_id"] for row in snapshots})
            == IDEAL_QUALIFICATION_SNAPSHOT_COUNT
            and not violations
        ),
        "confirmatory_seed_access_count": 0,
        "geometry_access_count": sum(
            int(row["firewall_audit"]["geometry_access_count"])
            for row in snapshots
        ),
        "lineage_violation_count": len(violations),
        "measurement_seed_access_count": 0,
        "qualification_lock": lock,
        "repeat_randomness_count": 0,
        "rng_instantiation_count": 0,
        "snapshot_count": len(snapshots),
        "violation_snapshot_ids": violations,
    }
    if report["IDEAL_GEOMETRY_QUALIFICATION_PASS"] is not True:
        raise ValueError("Development IDEAL geometry qualification failed")
    return snapshots, report


def build_independent_negative_controls(
    root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 21 Development-seed negatives and compare the Phase B baseline."""

    import numpy as np

    from .phase_b_snapshot_assets import (
        PHASE_B_CACHE_RELATIVE,
        PHASE_B_PLANNED_SNAPSHOTS_RELATIVE,
        PHASE_B_PLANNED_TRIALS_RELATIVE,
        PHASE_B_SNAPSHOT_LOCK_RELATIVE,
        read_phase_b_plans,
        read_phase_b_snapshot,
        validate_phase_b_snapshot_lock,
    )
    from .synthetic_confirmatory_v2_snapshot_builder import (
        DEVELOPMENT_GEOMETRY_SEEDS,
        build_development_independent_qualification_snapshot,
    )

    repository = Path(root).resolve()
    plans, trials = read_phase_b_plans(
        repository / PHASE_B_PLANNED_SNAPSHOTS_RELATIVE,
        repository / PHASE_B_PLANNED_TRIALS_RELATIVE,
    )
    lock = validate_phase_b_snapshot_lock(
        repository / PHASE_B_SNAPSHOT_LOCK_RELATIVE,
        repository / PHASE_B_CACHE_RELATIVE,
        plans,
    )
    lock_by_id = {row["snapshot_id"]: row for row in lock["snapshots"]}
    baselines = {
        (row["scene_variant"], int(row["geometry_seed_value"])): row
        for row in plans
        if row["condition"] == "INDEPENDENT_NOISE_FREE"
    }
    snapshots: list[dict[str, Any]] = []
    array_mismatch_count = 0
    checksum_mismatch_count = 0
    lineage_violation_count = 0
    firewall_violation_count = 0
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        for seed in DEVELOPMENT_GEOMETRY_SEEDS:
            geometry_seed = _assert_development_seed(
                seed, DEVELOPMENT_GEOMETRY_SEEDS, "INDEPENDENT geometry"
            )
            value = build_development_independent_qualification_snapshot(
                repository, scene=scene, geometry_seed=geometry_seed
            )
            plan = baselines[(scene, geometry_seed)]
            baseline = read_phase_b_snapshot(
                repository / PHASE_B_CACHE_RELATIVE,
                plan["snapshot_id"],
                expected_lock_entry=lock_by_id[plan["snapshot_id"]],
                arrays=True,
            )
            array_mismatches = [
                name
                for name in ("source", "target", "reference")
                if not np.array_equal(value[name], baseline[name])
            ]
            checksum_mismatches = [
                name
                for name in (
                    "source_checksum",
                    "target_checksum",
                    "reference_pose_checksum",
                )
                if value["metadata"][name] != baseline[name]
            ]
            metadata = value["metadata"]
            firewall = value["firewall_audit"]
            lineage_violation = bool(
                value["parent_indices"] is not None
                or metadata["source_has_target_parent_lineage"] is not False
                or metadata["source_is_target_subset"] is not False
                or metadata["source_parent_target_indices_sha256"] is not None
                or metadata["measurement_seed"] is not None
            )
            firewall_violation = bool(
                firewall["measurement_seed_access_count"] != 0
                or firewall["repeat_randomness_count"] != 0
                or firewall["rng_instantiation_count"] != 0
            )
            array_mismatch_count += len(array_mismatches)
            checksum_mismatch_count += len(checksum_mismatches)
            lineage_violation_count += int(lineage_violation)
            firewall_violation_count += int(firewall_violation)
            rows.append(
                {
                    "array_mismatch_fields": array_mismatches,
                    "baseline_snapshot_id": plan["snapshot_id"],
                    "checksum_mismatch_fields": checksum_mismatches,
                    "geometry_seed": geometry_seed,
                    "lineage_violation": lineage_violation,
                    "qualification_snapshot_id": metadata["snapshot_id"],
                    "scene_variant": scene,
                }
            )
            snapshots.append(value)
    qualification_lock = _qualification_snapshot_lock(
        snapshots, condition="INDEPENDENT_NOISE_FREE", lineage_expected=False
    )
    passed = bool(
        len(snapshots) == INDEPENDENT_NEGATIVE_SNAPSHOT_COUNT
        and len(trials) == 84
        and array_mismatch_count == 0
        and checksum_mismatch_count == 0
        and lineage_violation_count == 0
        and firewall_violation_count == 0
    )
    report = {
        "INDEPENDENT_NEGATIVE_CONTROL_PASS": passed,
        "array_mismatch_count": array_mismatch_count,
        "baseline_phase_b_snapshot_lock_sha256": file_sha256(
            repository / PHASE_B_SNAPSHOT_LOCK_RELATIVE
        ),
        "checksum_mismatch_count": checksum_mismatch_count,
        "confirmatory_seed_access_count": 0,
        "firewall_violation_count": firewall_violation_count,
        "lineage_violation_count": lineage_violation_count,
        "qualification_lock": qualification_lock,
        "rng_instantiation_count": 0,
        "rows": rows,
        "snapshot_count": len(snapshots),
    }
    if not passed:
        raise ValueError("Development INDEPENDENT negative-control qualification failed")
    return snapshots, report


def _load_backend_parameters(root: Path) -> tuple[dict[str, Any], Path]:
    contract_path = root / "frozen_assets/backend_parameter_contract.json"
    pcl_cli = root / "bin/pcl_point_to_plane_cli"
    if file_sha256(contract_path) != EXPECTED_BACKEND_PARAMETER_CONTRACT_SHA256:
        raise ValueError("backend parameter contract SHA changed")
    if file_sha256(pcl_cli) != EXPECTED_PCL_CLI_SHA256:
        raise ValueError("PCL CLI SHA changed")
    contract = _strict_object(contract_path, "backend parameter contract")
    for name, expected in (
        ("open3d", EXPECTED_OPEN3D_PARAMETER_SHA256),
        ("pcl", EXPECTED_PCL_PARAMETER_SHA256),
    ):
        section = contract.get(name)
        if (
            type(section) is not dict
            or section.get("canonical_sha256") != expected
            or canonical_json_sha256(section.get("parameters", {})) != expected
        ):
            raise ValueError(f"{name} frozen parameter contract changed")
    if contract.get("backend_parameter_difference_count") != 0:
        raise ValueError("backend parameter difference count is nonzero")
    return contract, pcl_cli


def _metric_gate(results: Sequence[Mapping[str, Any]], backend: str) -> dict[str, Any]:
    import numpy as np

    rows = [row for row in results if row["backend"] == backend]
    failures = sum(bool(row["solver_failure"]) for row in rows)
    nonfinite = sum(not bool(row["finite_output"]) for row in rows)
    translations = [
        float(row["translation_update_m"])
        for row in rows
        if row["translation_update_m"] is not None
    ]
    rotations = [
        float(row["rotation_update_rad"])
        for row in rows
        if row["rotation_update_rad"] is not None
    ]
    translation_q95 = (
        None
        if len(translations) != IDEAL_QUALIFICATION_SNAPSHOT_COUNT
        else float(np.quantile(translations, 0.95, method=QUANTILE_METHOD))
    )
    rotation_q95 = (
        None
        if len(rotations) != IDEAL_QUALIFICATION_SNAPSHOT_COUNT
        else float(np.quantile(rotations, 0.95, method=QUANTILE_METHOD))
    )
    passed = bool(
        len(rows) == IDEAL_QUALIFICATION_SNAPSHOT_COUNT
        and failures == 0
        and nonfinite == 0
        and translation_q95 is not None
        and translation_q95 <= TRANSLATION_THRESHOLD_M
        and rotation_q95 is not None
        and rotation_q95 <= ROTATION_THRESHOLD_RAD
    )
    return {
        "QUALIFICATION_BACKEND_GATE_PASS": passed,
        "backend": backend,
        "finite_metric_count": len(translations),
        "nonfinite_output_count": nonfinite,
        "quantile_method": QUANTILE_METHOD,
        "rotation_q95_rad": rotation_q95,
        "rotation_threshold_rad": ROTATION_THRESHOLD_RAD,
        "solver_failure_count": failures,
        "translation_q95_m": translation_q95,
        "translation_threshold_m": TRANSLATION_THRESHOLD_M,
        "trial_count": len(rows),
    }


def run_ideal_backend_controls(
    root: str | Path,
    snapshots: Sequence[Mapping[str, Any]],
    *,
    qualification_lock_sha256: str,
    workers: int = 2,
) -> dict[str, Any]:
    """Run the unchanged Open3D/PCL adapters over 21 qualification controls."""

    if len(snapshots) != IDEAL_QUALIFICATION_SNAPSHOT_COUNT:
        raise ValueError("IDEAL backend qualification requires exactly 21 snapshots")
    if workers not in (1, 2):
        raise ValueError("qualification workers must be one or two")
    repository = Path(root).resolve()
    contract, pcl_cli = _load_backend_parameters(repository)
    binding = qualification_execution_binding(repository)
    from .full_synthetic_snapshot_builder import DEVELOPMENT_PROTOCOL_SHA256
    from .phase_a_execution_chain_audit import (
        execute_open3d_fixture,
        execute_pcl_fixture,
    )
    from .phase_a_execution_chain_fixture import FixtureSnapshot
    from .phase_a_trial_result_schema import OPEN3D_BACKEND, PCL_BACKEND

    tasks: list[tuple[Mapping[str, Any], str]] = [
        (snapshot, backend)
        for snapshot in snapshots
        for backend in (OPEN3D_BACKEND, PCL_BACKEND)
    ]

    def execute(task: tuple[Mapping[str, Any], str]) -> dict[str, Any]:
        snapshot, backend = task
        metadata = snapshot["metadata"]
        checksums = {
            "reference_pose_checksum": metadata["reference_pose_checksum"],
            "snapshot_checksum": metadata["snapshot_checksum"],
            "source_checksum": metadata["source_checksum"],
            "target_checksum": metadata["target_checksum"],
        }
        fixture = FixtureSnapshot(
            snapshot_id=metadata["snapshot_id"],
            scene_variant=metadata["scene_variant"],
            condition="IDEAL_MATCHED",
            source=snapshot["source"],
            target=snapshot["target"],
            reference=snapshot["reference"],
            expected_failure_classifications=("NONE",),
            checksums=checksums,
        )
        common = {
            "backend": backend,
            "condition": "IDEAL_MATCHED",
            "implementation_sha256": binding["qualification_binding_sha256"],
            "planned_trial_id": f"{metadata['snapshot_id']}/{backend}",
            "protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
            "reference_pose_checksum": checksums["reference_pose_checksum"],
            "scene_variant": metadata["scene_variant"],
            "schema_version": "phase_a_trial_result_v1",
            "snapshot_checksum": checksums["snapshot_checksum"],
            "snapshot_id": metadata["snapshot_id"],
            "snapshot_lock_sha256": qualification_lock_sha256,
            "source_checksum": checksums["source_checksum"],
            "target_checksum": checksums["target_checksum"],
        }
        if backend == OPEN3D_BACKEND:
            return execute_open3d_fixture(
                fixture=fixture,
                common=common,
                parameters=contract["open3d"]["parameters"],
            )
        return execute_pcl_fixture(
            fixture=fixture,
            common=common,
            parameters=contract["pcl"]["parameters"],
            pcl_cli=pcl_cli,
        )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(pending):
            results.append(future.result())
    results.sort(key=lambda row: row["planned_trial_id"])
    gates = {
        backend: _metric_gate(results, backend)
        for backend in (OPEN3D_BACKEND, PCL_BACKEND)
    }
    scientific_projection = [
        {key: value for key, value in row.items() if key != "runtime_ms"}
        for row in results
    ]
    passed = bool(
        len(results) == IDEAL_QUALIFICATION_TRIAL_COUNT
        and len({row["planned_trial_id"] for row in results})
        == IDEAL_QUALIFICATION_TRIAL_COUNT
        and all(gate["QUALIFICATION_BACKEND_GATE_PASS"] for gate in gates.values())
    )
    report = {
        "IDEAL_DUAL_BACKEND_CONTROL_PASS": passed,
        "backend_execution_count": len(results),
        "backend_gates": gates,
        "formal_v2_backend_execution_count": 0,
        "native_execution_count": 0,
        "qualification_result_lock_sha256": canonical_json_sha256(
            {"results": scientific_projection}
        ),
        "results": results,
        "trial_count": len(results),
    }
    if not passed:
        raise RuntimeError("IDEAL qualification backend controls failed")
    return report


def _independent_development_record_reference(
    plan: Mapping[str, str], protocol: Any
) -> dict[str, Any]:
    """Rebuild only the frozen Development record oracle.

    The candidate under test is the public v2 Development helper.  This
    separate call through the frozen Development entry point supplies an
    independently instantiated firewall/bundle for exact realized-record
    comparison; persisted arrays remain the primary historical baseline.
    """

    from .full_synthetic_development_protocol import (
        GEOMETRY_SEEDS,
        MEASUREMENT_SEEDS,
    )
    from .phase_b_generator import (
        DevelopmentSeedFirewall,
        SnapshotKey,
        build_snapshot,
    )

    geometry_seed = _assert_development_seed(
        int(plan["geometry_seed_value"]), GEOMETRY_SEEDS, "record-oracle geometry"
    )
    measurement_seed = _assert_development_seed(
        int(plan["measurement_seed_value"]),
        MEASUREMENT_SEEDS,
        "record-oracle measurement",
    )
    firewall = DevelopmentSeedFirewall(protocol)
    bundle = build_snapshot(
        protocol,
        firewall,
        SnapshotKey(
            scene_variant=str(plan["scene_variant"]),
            geometry_seed=geometry_seed,
            measurement_seed=measurement_seed,
            repeat_index=int(plan["repeat_index"]),
            noise_condition=str(plan["condition"]),
        ),
    )
    return {
        "firewall_audit": firewall.report(),
        "map_point_count_before_dropout": int(bundle.map_point_count_before_dropout),
        "record_checksums": dict(bundle.checksums),
        "scan_point_count_before_dropout": int(bundle.scan_point_count_before_dropout),
    }


def regress_nonideal_development_payloads(
    root: str | Path, *, workers: int = 2
) -> dict[str, Any]:
    """Rebuild and compare all 1,050 frozen non-IDEAL Development snapshots."""

    import numpy as np

    from .full_synthetic_development_protocol import (
        NEW_CONDITIONS,
        SNAPSHOT_CACHE_RELATIVE,
        SNAPSHOT_LOCK_RELATIVE,
        read_full_synthetic_plans,
    )
    from .full_synthetic_snapshot_builder import (
        SNAPSHOT_BUILDER_SHA256,
        load_full_synthetic_generator_protocol,
        read_full_synthetic_snapshot,
        validate_full_synthetic_snapshot_lock,
    )
    from .synthetic_confirmatory_v2_snapshot_builder import (
        build_development_nonideal_regression_snapshot,
    )

    if workers not in (1, 2):
        raise ValueError("Development regression workers must be one or two")
    repository = Path(root).resolve()
    plans = read_full_synthetic_plans(repository)[0]
    if len(plans) != DEVELOPMENT_NONIDEAL_SNAPSHOT_COUNT:
        raise ValueError("Development regression plan is not exactly 1,050 snapshots")
    cache = repository / SNAPSHOT_CACHE_RELATIVE
    lock_path = repository / SNAPSHOT_LOCK_RELATIVE
    lock = validate_full_synthetic_snapshot_lock(lock_path, cache, plans)
    lock_by_id = {row["snapshot_id"]: row for row in lock["snapshots"]}
    reference_protocol = load_full_synthetic_generator_protocol(repository)

    def compare(plan: Mapping[str, str]) -> dict[str, Any]:
        actual = read_full_synthetic_snapshot(
            cache,
            plan,
            expected_lock_entry=lock_by_id[str(plan["snapshot_id"])],
            arrays=True,
        )
        # This is the public Development-only entry into the exact shared v2
        # non-IDEAL numerical helper.  The formal builder uses that same helper,
        # but formal v2 plans/seeds are rejected throughout this qualification.
        candidate = build_development_nonideal_regression_snapshot(repository, plan)
        record_reference = _independent_development_record_reference(
            plan, reference_protocol
        )
        array_mismatches = [
            name
            for name in ("source", "target", "reference")
            if not np.array_equal(actual[name], candidate[name])
        ]
        expected_projection = {
            name: actual["metadata"][name] for name in _SCIENTIFIC_METADATA_FIELDS
        }
        candidate_metadata = {
            **candidate["metadata"],
            "geometry_seed_index": int(plan["geometry_seed_index"]),
            "measurement_seed_index": int(plan["measurement_seed_index"]),
            # The shared v2 helper calls the exact frozen numerical builder;
            # its byte identity is bound independently from the v2 envelope.
            "snapshot_builder_sha256": SNAPSHOT_BUILDER_SHA256,
        }
        candidate_projection = {
            name: candidate_metadata[name] for name in _SCIENTIFIC_METADATA_FIELDS
        }
        metadata_mismatch_fields = sorted(
            name
            for name in _SCIENTIFIC_METADATA_FIELDS
            if expected_projection[name] != candidate_projection[name]
        )
        checksum_mismatch_fields = sorted(
            name
            for name in (
                "source_checksum",
                "target_checksum",
                "reference_pose_checksum",
            )
            if actual["metadata"][name] != candidate["metadata"][name]
        )
        aggregate_snapshot_checksum_equal = bool(
            actual["metadata"]["snapshot_checksum"]
            == candidate["metadata"]["snapshot_checksum"]
        )
        persisted_baseline_records = {
            name: actual["metadata"].get(name)
            for name in ("noise_checksum", "dropout_checksum")
        }
        candidate_records = dict(
            candidate.get(
                "record_checksums",
                {"dropout_checksum": None, "noise_checksum": None},
            )
        )
        persisted_record_mismatch_fields = sorted(
            name
            for name in persisted_baseline_records
            if persisted_baseline_records[name] is not None
            and persisted_baseline_records[name] != candidate_records[name]
        )
        reference_records = dict(record_reference["record_checksums"])
        record_oracle_mismatch_fields = sorted(
            name
            for name in set(reference_records) | set(candidate_records)
            if reference_records.get(name) != candidate_records.get(name)
        )
        missing_candidate_records = sorted(
            name for name, value in candidate_records.items() if value is None
        )
        firewall = candidate["firewall_audit"]
        reference_firewall = record_reference["firewall_audit"]
        firewall_violation = any(
            audit.get(name) != 0
            for audit in (firewall, reference_firewall)
            for name in (
                "CONFIRMATORY_SEED_INSTANTIATION_COUNT",
                "GT_OPTIMIZATION_LEAKAGE_COUNT",
                "OLD_CAPTURE_RANGE_TEST_SEED_ACCESS_COUNT",
            )
        )
        before_dropout_count_mismatch = bool(
            candidate.get("scan_point_count_before_dropout")
            != record_reference["scan_point_count_before_dropout"]
            or candidate.get("map_point_count_before_dropout")
            != record_reference["map_point_count_before_dropout"]
        )
        return {
            "array_mismatch_fields": array_mismatches,
            "aggregate_snapshot_checksum_equal": aggregate_snapshot_checksum_equal,
            "candidate_dropout_checksum": candidate_records["dropout_checksum"],
            "candidate_noise_checksum": candidate_records["noise_checksum"],
            "candidate_record_missing_fields": missing_candidate_records,
            "checksum_mismatch_fields": checksum_mismatch_fields,
            "condition": str(plan["condition"]),
            "firewall_violation": firewall_violation,
            "geometry_seed": int(plan["geometry_seed_value"]),
            "measurement_seed": int(plan["measurement_seed_value"]),
            "metadata_mismatch_fields": metadata_mismatch_fields,
            "before_dropout_count_mismatch": before_dropout_count_mismatch,
            "record_baseline_available_fields": sorted(
                name
                for name, value in persisted_baseline_records.items()
                if value is not None
            ),
            "persisted_record_mismatch_fields": persisted_record_mismatch_fields,
            "record_oracle_mismatch_fields": record_oracle_mismatch_fields,
            "repeat_index": int(plan["repeat_index"]),
            "reference_rng_construction_count": int(
                reference_firewall["RNG_CONSTRUCTION_COUNT"]
            ),
            "v2_candidate_rng_construction_count": int(
                firewall["RNG_CONSTRUCTION_COUNT"]
            ),
            "scene_variant": str(plan["scene_variant"]),
            "scientific_payload_sha256": canonical_json_sha256(
                {
                    "metadata": candidate_projection,
                    "record_checksums": candidate_records,
                }
            ),
            "snapshot_id": str(plan["snapshot_id"]),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(compare, plan): plan for plan in plans}
        for future in as_completed(pending):
            rows.append(future.result())
    rows.sort(key=lambda row: row["snapshot_id"])
    condition_counts = Counter(row["condition"] for row in rows)
    array_mismatches = sum(len(row["array_mismatch_fields"]) for row in rows)
    metadata_mismatches = sum(len(row["metadata_mismatch_fields"]) for row in rows)
    checksum_mismatches = sum(len(row["checksum_mismatch_fields"]) for row in rows)
    aggregate_envelope_differences = sum(
        not row["aggregate_snapshot_checksum_equal"] for row in rows
    )
    persisted_record_mismatches = sum(
        len(row["persisted_record_mismatch_fields"]) for row in rows
    )
    record_oracle_mismatches = sum(
        len(row["record_oracle_mismatch_fields"]) for row in rows
    )
    before_dropout_count_mismatches = sum(
        bool(row["before_dropout_count_mismatch"]) for row in rows
    )
    candidate_record_missing = sum(
        len(row["candidate_record_missing_fields"]) for row in rows
    )
    baseline_record_available = sum(
        len(row["record_baseline_available_fields"]) for row in rows
    )
    firewall_violations = sum(bool(row["firewall_violation"]) for row in rows)
    candidate_rng_count = sum(
        int(row["v2_candidate_rng_construction_count"]) for row in rows
    )
    reference_rng_count = sum(
        int(row["reference_rng_construction_count"]) for row in rows
    )
    plan_identity_fields = frozenset(
        {
            "condition",
            "geometry_seed",
            "geometry_seed_index",
            "measurement_seed",
            "measurement_seed_index",
            "repeat_index",
            "scene_variant",
            "snapshot_id",
        }
    )

    def row_record_changes(row: Mapping[str, Any]) -> set[str]:
        return set(row["persisted_record_mismatch_fields"]) | set(
            row["record_oracle_mismatch_fields"]
        ) | set(row["candidate_record_missing_fields"])

    def row_scientific_change(row: Mapping[str, Any]) -> bool:
        return bool(
            row["array_mismatch_fields"]
            or row["checksum_mismatch_fields"]
            or row["metadata_mismatch_fields"]
            or row_record_changes(row)
            or row["before_dropout_count_mismatch"]
        )

    source_checksum_changes = sum(
        "source_checksum" in row["checksum_mismatch_fields"]
        or "source" in row["array_mismatch_fields"]
        for row in rows
    )
    target_checksum_changes = sum(
        "target_checksum" in row["checksum_mismatch_fields"]
        or "target" in row["array_mismatch_fields"]
        for row in rows
    )
    reference_checksum_changes = sum(
        "reference_pose_checksum" in row["checksum_mismatch_fields"]
        or "reference" in row["array_mismatch_fields"]
        for row in rows
    )
    noise_record_changes = sum(
        "noise_checksum" in row_record_changes(row) for row in rows
    )
    dropout_record_changes = sum(
        "dropout_checksum" in row_record_changes(row) for row in rows
    )
    plan_identity_changes = sum(
        bool(plan_identity_fields & set(row["metadata_mismatch_fields"]))
        for row in rows
    )
    scientific_payload_changes = sum(row_scientific_change(row) for row in rows)
    metadata_only_changes = sum(
        not row["aggregate_snapshot_checksum_equal"]
        and not row_scientific_change(row)
        for row in rows
    )
    expected_rng_count = 1260
    record_lock_core = {
        "entries": [
            {
                "dropout_checksum": row["candidate_dropout_checksum"],
                "noise_checksum": row["candidate_noise_checksum"],
                "scientific_payload_sha256": row["scientific_payload_sha256"],
                "snapshot_id": row["snapshot_id"],
            }
            for row in rows
        ],
        "schema_version": "development_nonideal_scientific_payload_lock_v1",
    }
    passed = bool(
        len(rows) == DEVELOPMENT_NONIDEAL_SNAPSHOT_COUNT
        and condition_counts == Counter({name: 210 for name in NEW_CONDITIONS})
        and array_mismatches == 0
        and metadata_mismatches == 0
        and checksum_mismatches == 0
        and persisted_record_mismatches == 0
        and record_oracle_mismatches == 0
        and before_dropout_count_mismatches == 0
        and candidate_record_missing == 0
        and firewall_violations == 0
        and candidate_rng_count == expected_rng_count
        and reference_rng_count == expected_rng_count
        and source_checksum_changes == 0
        and target_checksum_changes == 0
        and reference_checksum_changes == 0
        and noise_record_changes == 0
        and dropout_record_changes == 0
        and plan_identity_changes == 0
        and scientific_payload_changes == 0
    )
    report = {
        "DEVELOPMENT_NONIDEAL_SCIENTIFIC_PAYLOAD_REGRESSION_PASS": passed,
        "METADATA_ONLY_CHANGE_COUNT": metadata_only_changes,
        "NONIDEAL_DROPOUT_RECORD_CHANGE_COUNT": dropout_record_changes,
        "NONIDEAL_NOISE_RECORD_CHANGE_COUNT": noise_record_changes,
        "NONIDEAL_PLAN_IDENTITY_CHANGE_COUNT": plan_identity_changes,
        "NONIDEAL_REFERENCE_CHECKSUM_CHANGE_COUNT": reference_checksum_changes,
        "NONIDEAL_SCIENTIFIC_PAYLOAD_CHANGE_COUNT": scientific_payload_changes,
        "NONIDEAL_SOURCE_CHECKSUM_CHANGE_COUNT": source_checksum_changes,
        "NONIDEAL_TARGET_CHECKSUM_CHANGE_COUNT": target_checksum_changes,
        "SCIENTIFIC_PAYLOAD_CHANGE_COUNT": scientific_payload_changes,
        "V2_NONIDEAL_SHARED_PATH_REGRESSION_PASS": passed,
        "array_mismatch_count": array_mismatches,
        "baseline_noise_dropout_record_available_field_count": (
            baseline_record_available
        ),
        "candidate_noise_dropout_record_count": 2 * len(rows),
        "candidate_noise_dropout_record_missing_count": candidate_record_missing,
        "candidate_v2_helper_call_count": len(rows),
        "candidate_v2_helper_callable": (
            "phase_a_harness.synthetic_confirmatory_v2_snapshot_builder."
            "build_development_nonideal_regression_snapshot"
        ),
        "checksum_mismatch_count": checksum_mismatches,
        "condition_snapshot_counts": dict(sorted(condition_counts.items())),
        "confirmatory_seed_access_count": 0,
        "before_dropout_count_mismatch_count": before_dropout_count_mismatches,
        "development_record_oracle_rng_construction_count": reference_rng_count,
        "expected_development_rng_construction_count": expected_rng_count,
        "formal_v2_rng_instantiation_count": 0,
        "frozen_snapshot_lock_file_sha256": file_sha256(lock_path),
        "metadata_envelope_fields_excluded_from_scientific_comparison": list(
            _METADATA_ENVELOPE_FIELDS
        ),
        "metadata_only_v1_v2_snapshot_checksum_difference_count": (
            aggregate_envelope_differences
        ),
        "metadata_mismatch_count": metadata_mismatches,
        "noise_dropout_record_mismatch_count_where_persisted_baseline_available": (
            persisted_record_mismatches
        ),
        "record_oracle_mismatch_count": record_oracle_mismatches,
        "record_availability_note": (
            "Historical Development metadata does not persist realized noise/dropout "
            "record checksums. Candidate record checksums are therefore compared to "
            "an independently instantiated frozen-Development record oracle and bound "
            "below; persisted-field comparisons are additionally enforced wherever a "
            "baseline field exists, while exact output-array equivalence remains "
            "mandatory for every snapshot."
        ),
        "snapshot_checksum_envelope_note": (
            "The v2 aggregate checksum binds an explicit null parent-lineage field "
            "for non-IDEAL snapshots while the Development v1 aggregate omitted that "
            "field. This expected envelope-only difference is not a scientific "
            "payload difference; all three raw array checksums remain mandatory."
        ),
        "rows": rows,
        "schema_version": DEVELOPMENT_REGRESSION_SCHEMA,
        "scientific_payload_lock": {
            **record_lock_core,
            "scientific_payload_lock_sha256": canonical_json_sha256(record_lock_core),
        },
        "security_firewall_violation_count": firewall_violations,
        "snapshot_count": len(rows),
        "shared_numerical_helper": "_nonideal_arrays",
        "v2_candidate_rng_construction_count": candidate_rng_count,
    }
    if not passed:
        raise RuntimeError("1,050-snapshot Development scientific regression failed")
    return report


def fixture_chain_hooks(root: str | Path) -> dict[str, Any]:
    """Describe the existing full 3/6 fixture chain without executing it."""

    repository = Path(root).resolve()
    manifest = repository / "frozen_assets/frozen_experiment_manifest.json"
    dependencies = {
        "fixture_plan": repository / "frozen_assets/fixtures/fixture_plan.json",
        "fixture_snapshot_lock": (
            repository / "frozen_assets/fixtures/fixture_snapshot_lock.json"
        ),
        "fixture_parameter_lock": (
            repository / "frozen_assets/fixtures/fixture_backend_parameter_lock.json"
        ),
        "fixture_qualification": (
            repository / "src/phase_a_harness/fixture_qualification.py"
        ),
        "fixture_publisher": repository / "src/phase_a_harness/fixture_publication.py",
        "fixture_artifact_verifier": (
            repository
            / "src/phase_a_harness/fixture_publication_artifact_verifier.py"
        ),
        "manifest": manifest,
    }
    if any(not path.is_file() for path in dependencies.values()):
        missing = [name for name, path in dependencies.items() if not path.is_file()]
        raise FileNotFoundError(f"fixture-chain dependency missing: {missing}")
    return {
        "EXECUTION_CHAIN_FIXTURE_HOOK_READY": True,
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        "callable": "phase_a_harness.fixture_qualification.qualify_fixtures",
        "dependency_sha256": {
            name: file_sha256(path) for name, path in sorted(dependencies.items())
        },
        "expected_backend_execution_count": 6,
        "expected_snapshot_count": 3,
        "expected_trial_count": 6,
        "fixture_label": "FIXTURE AUDIT -- NOT SCIENTIFIC DATA",
        "invocation_template": (
            "PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX=/home/lj/.local/share/"
            "degen-lio-micromamba /home/lj/.local/bin/micromamba run "
            "-n degen-lio-zprm-py311 python scripts/run_fixture_qualification.py "
            "--manifest frozen_assets/frozen_experiment_manifest.json "
            "--output-dir <NEW_ABSENT_QUALIFICATION_DIRECTORY>"
        ),
        "schema_version": "synthetic_confirmatory_v2_fixture_chain_hook_v1",
    }


def run_seed_free_fixture_chain(
    root: str | Path, *, output_dir: str | Path
) -> dict[str, Any]:
    """Execute the isolated 3/6 fixture chain at a new non-formal path."""

    repository = Path(root).resolve()
    destination = Path(output_dir).resolve()
    forbidden = (
        repository / "results/synthetic_confirmatory_v2",
        repository / "data/synthetic_confirmatory_v2_snapshots",
        repository / "artifacts/synthetic_confirmatory_v2",
        repository / "artifacts/synthetic_confirmatory_v2_prerun",
    )
    if any(_inside(destination, path) or _inside(path, destination) for path in forbidden):
        raise PermissionError("fixture output overlaps a formal v2 path")
    if destination.exists():
        raise FileExistsError(f"fixture output already exists: {destination}")
    hooks = fixture_chain_hooks(repository)
    from .fixture_qualification import qualify_fixtures

    report = qualify_fixtures(
        manifest_path=repository / "frozen_assets/frozen_experiment_manifest.json",
        output_dir=destination,
    )
    passed = bool(
        report.get("FIXTURE_QUALIFICATION_PASS") is True
        and report.get("fixture_snapshot_count") == 3
        and report.get("fixture_trial_count") == 6
        and report.get("backend_execution_count") == 6
        and report.get("resume_backend_execution_count") == 0
        and report.get("fresh_resume_scientific_equivalence") is True
        and report.get("analysis_verifier_difference_count") == 0
        and report.get("publisher_pass") is True
        and report.get("artifact_verifier_pass") is True
    )
    if not passed:
        raise RuntimeError("seed-free 3/6 fixture chain failed")
    return {
        **hooks,
        "EXECUTION_CHAIN_FIXTURE_PASS": True,
        "fixture_output_dir": str(destination),
        "fixture_report": report,
        "qualification_backend_execution_count": 6,
    }


def execute_bounded_qualification(
    root: str | Path,
    *,
    workers: int = 2,
    fixture_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all bounded checks while proving formal-v2 access remains zero."""

    repository = Path(root).resolve()
    qualification_binding_start = qualification_execution_binding(repository)
    formal_counters = _zero_formal_counters()
    monitor = QualificationAccessMonitor(repository)
    monitor.install()
    ideal_snapshots, ideal = build_ideal_geometry_qualification_snapshots(repository)
    independent_snapshots, independent = build_independent_negative_controls(repository)
    # Retain no ambiguity: the negative controls are constructed and checked,
    # but are never sent to a backend in this bounded qualification.
    del independent_snapshots
    controls = run_ideal_backend_controls(
        repository,
        ideal_snapshots,
        qualification_lock_sha256=ideal["qualification_lock"][
            "qualification_lock_sha256"
        ],
        workers=workers,
    )
    development = regress_nonideal_development_payloads(repository, workers=workers)
    fixture = (
        fixture_chain_hooks(repository)
        if fixture_output_dir is None
        else run_seed_free_fixture_chain(repository, output_dir=fixture_output_dir)
    )
    _assert_zero_formal_counters(formal_counters)
    access = monitor.report()
    from .asset_verifier import source_runtime_import_paths

    source_imports = source_runtime_import_paths()
    qualification_binding_end = qualification_execution_binding(repository)
    qualification_binding_stable = bool(
        qualification_binding_start == qualification_binding_end
        and all(
            row.get("implementation_sha256")
            == qualification_binding_start["qualification_binding_sha256"]
            for row in controls["results"]
        )
    )
    access_pass = bool(
        not source_imports
        and all(value == 0 for value in access.values())
        and all(value == 0 for value in formal_counters.values())
    )
    fixture_satisfied = bool(
        fixture.get("EXECUTION_CHAIN_FIXTURE_PASS") is True
        or fixture.get("EXECUTION_CHAIN_FIXTURE_HOOK_READY") is True
    )
    passed = bool(
        ideal["IDEAL_GEOMETRY_QUALIFICATION_PASS"] is True
        and independent["INDEPENDENT_NEGATIVE_CONTROL_PASS"] is True
        and controls["IDEAL_DUAL_BACKEND_CONTROL_PASS"] is True
        and development[
            "DEVELOPMENT_NONIDEAL_SCIENTIFIC_PAYLOAD_REGRESSION_PASS"
        ]
        is True
        and fixture_satisfied
        and access_pass
        and qualification_binding_stable
    )
    return {
        "BOUNDED_V2_QUALIFICATION_PASS": passed,
        "FORMAL_CONFIRMATORY_SCIENCE_EVALUATED": False,
        "FORMAL_V2_ACCESS_ZERO_PASS": access_pass,
        "QUALIFICATION_EXECUTION_BINDING_STABLE_PASS": (
            qualification_binding_stable
        ),
        "access_monitor": access,
        "development_nonideal_regression": development,
        "fixture_chain": fixture,
        "formal_v2_access_counters": formal_counters,
        "ideal_backend_controls": controls,
        "ideal_geometry_qualification": ideal,
        "independent_negative_controls": independent,
        "qualification_only": True,
        "qualification_execution_binding": qualification_binding_start,
        "qualification_execution_binding_end_sha256": qualification_binding_end[
            "qualification_binding_sha256"
        ],
        "schema_version": QUALIFICATION_SCHEMA,
        "source_repository_runtime_import_count": len(source_imports),
        "source_repository_runtime_import_paths": source_imports,
    }


__all__ = [
    "DEVELOPMENT_NONIDEAL_SNAPSHOT_COUNT",
    "FORMAL_ZERO_COUNTERS",
    "IDEAL_QUALIFICATION_SNAPSHOT_COUNT",
    "IDEAL_QUALIFICATION_TRIAL_COUNT",
    "INDEPENDENT_NEGATIVE_SNAPSHOT_COUNT",
    "QUALIFICATION_SCHEMA",
    "QualificationAccessMonitor",
    "build_ideal_geometry_qualification_snapshots",
    "build_independent_negative_controls",
    "execute_bounded_qualification",
    "fixture_chain_hooks",
    "qualification_execution_binding",
    "qualification_execution_plan",
    "regress_nonideal_development_payloads",
    "run_ideal_backend_controls",
    "run_seed_free_fixture_chain",
]
