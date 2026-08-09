"""Deterministic seed-free fixtures for the Phase A execution-chain audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .backend_phase_a_metrics import canonical_backend_inputs
from .phase_a_trial_result_schema import canonical_json_sha256, file_sha256


FIXTURE_PLAN_RELATIVE = Path("tests/data/phase_a_execution_chain_audit/fixture_plan.json")
FIXTURE_LOCK_RELATIVE = Path(
    "tests/data/phase_a_execution_chain_audit/fixture_snapshot_lock.json"
)
FIXTURE_PARAMETER_LOCK_RELATIVE = Path(
    "tests/data/phase_a_execution_chain_audit/fixture_backend_parameter_lock.json"
)
FORMAL_CACHE_RELATIVE = Path("data/zero_perturbation/backend_phase_a_v1_2_stage0")


@dataclass(frozen=True)
class FixtureSnapshot:
    snapshot_id: str
    scene_variant: str
    condition: str
    source: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    expected_failure_classifications: tuple[str, ...]
    checksums: Mapping[str, str]


def _axis(start: float, stop: float, count: int) -> list[float]:
    return [float(value) for value in np.linspace(start, stop, count)]


def _base_points() -> np.ndarray:
    points: set[tuple[float, float, float]] = set()
    grid = _axis(-1.0, 1.0, 13)
    points.update((0.0, y, z) for y in grid for z in grid)
    points.update((x, 0.0, z) for x in grid for z in grid)
    points.update((x, y, 0.0) for x in grid for y in grid)

    # Five exposed faces of one off-centre cuboid bump.
    xs = _axis(0.19, 0.55, 9)
    ys = _axis(-0.44, -0.18, 9)
    zs = _axis(0.0, 0.36, 9)
    points.update((x, y, 0.36) for x in xs for y in ys)
    points.update((0.19, y, z) for y in ys for z in zs)
    points.update((0.55, y, z) for y in ys for z in zs)
    points.update((x, -0.44, z) for x in xs for z in zs)
    points.update((x, -0.18, z) for x in xs for z in zs)
    result = np.asarray(sorted(points), dtype="<f4")
    if len(result) < 300 or len(np.unique(result, axis=0)) != len(result):
        raise RuntimeError("fixture geometry construction failed")
    if np.linalg.matrix_rank(np.cov(result.astype(np.float64).T)) != 3:
        raise RuntimeError("fixture geometry is not full-rank")
    result.setflags(write=False)
    return result


def _rpy_degrees(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx, ry, rz = [math.radians(value) for value in (roll, pitch, yaw)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def _transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    result = np.eye(4, dtype="<f8")
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(tuple(translation), dtype=np.float64)
    result.setflags(write=False)
    return result


def _snapshot(
    *,
    snapshot_id: str,
    scene_variant: str,
    condition: str,
    source: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    expected: tuple[str, ...],
) -> FixtureSnapshot:
    canonical = canonical_backend_inputs(source, target, reference)
    components = {
        "source_checksum": canonical["source_checksum"],
        "target_checksum": canonical["target_checksum"],
        "reference_pose_checksum": canonical["reference_pose_checksum"],
    }
    snapshot_checksum = canonical_json_sha256(
        {
            **components,
            "snapshot_id": snapshot_id,
            "scene_variant": scene_variant,
            "condition": condition,
        }
    )
    return FixtureSnapshot(
        snapshot_id=snapshot_id,
        scene_variant=scene_variant,
        condition=condition,
        source=canonical["source_points"],
        target=canonical["target_points"],
        reference=canonical["reference_pose"],
        expected_failure_classifications=expected,
        checksums={**components, "snapshot_checksum": snapshot_checksum},
    )


def build_fixture_snapshots() -> tuple[FixtureSnapshot, ...]:
    base = _base_points()
    identity = np.eye(4, dtype="<f8")
    reference = _transform(_rpy_degrees(1.0, -1.5, 2.0), (0.08, -0.04, 0.06))
    transformed = (
        (reference[:3, :3] @ base.astype(np.float64).T).T + reference[:3, 3]
    ).astype("<f4")
    separated = (base.astype(np.float64) + np.array([4.0, -3.0, 2.5])).astype("<f4")
    return (
        _snapshot(
            snapshot_id="fixture-audit-v1/identity",
            scene_variant="AUDIT_ASYMMETRIC_3D",
            condition="FIXTURE_IDENTITY",
            source=base,
            target=base,
            reference=identity,
            expected=("NONE",),
        ),
        _snapshot(
            snapshot_id="fixture-audit-v1/nonidentity-reference",
            scene_variant="AUDIT_ASYMMETRIC_3D",
            condition="FIXTURE_NONIDENTITY_REFERENCE",
            source=base,
            target=transformed,
            reference=reference,
            expected=("NONE",),
        ),
        _snapshot(
            snapshot_id="fixture-audit-v1/no-correspondence",
            scene_variant="AUDIT_SEPARATED_ASYMMETRIC_3D",
            condition="FIXTURE_NO_CORRESPONDENCE",
            source=base,
            target=separated,
            reference=identity,
            expected=("NO_CORRESPONDENCES", "SCIENTIFIC_SOLVER_FAILURE"),
        ),
    )


def fixture_plan_payload() -> dict[str, Any]:
    snapshots = build_fixture_snapshots()
    return {
        "fixture_only": True,
        "formal_phase_a": False,
        "random_seed_used": False,
        "schema_version": "phase_a_execution_chain_fixture_plan_v1",
        "snapshots": [
            {
                "condition": item.condition,
                "expected_failure_classifications": list(
                    item.expected_failure_classifications
                ),
                "scene_variant": item.scene_variant,
                "snapshot_id": item.snapshot_id,
            }
            for item in snapshots
        ],
    }


def fixture_lock_payload(plan_sha256: str) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "backend_parameter_lock_path": FIXTURE_PARAMETER_LOCK_RELATIVE.as_posix(),
        "backend_parameter_lock_sha256": file_sha256(
            repository / FIXTURE_PARAMETER_LOCK_RELATIVE
        ),
        "fixture_only": True,
        "fixture_plan_sha256": plan_sha256,
        "formal_phase_a": False,
        "formal_seed_values_included": False,
        "random_seed_used": False,
        "schema_version": "phase_a_execution_chain_fixture_snapshot_lock_v1",
        "snapshots": [
            {
                "condition": item.condition,
                "expected_failure_classifications": list(
                    item.expected_failure_classifications
                ),
                "reference_pose_checksum": item.checksums["reference_pose_checksum"],
                "reference_pose_4x4": item.reference.tolist(),
                "scene_variant": item.scene_variant,
                "snapshot_checksum": item.checksums["snapshot_checksum"],
                "snapshot_id": item.snapshot_id,
                "source_checksum": item.checksums["source_checksum"],
                "target_checksum": item.checksums["target_checksum"],
            }
            for item in build_fixture_snapshots()
        ],
    }


def validate_fixture_lock(root: str | Path, lock_path: str | Path) -> tuple[dict[str, Any], tuple[FixtureSnapshot, ...]]:
    repository = Path(root).resolve()
    candidate = Path(lock_path).resolve()
    formal = (repository / FORMAL_CACHE_RELATIVE).resolve()
    if candidate == formal or formal in candidate.parents:
        raise PermissionError("formal Stage-0 cache is forbidden")
    import json

    value = json.loads(candidate.read_text(encoding="utf-8"))
    plan_path = repository / FIXTURE_PLAN_RELATIVE
    if value != fixture_lock_payload(file_sha256(plan_path)):
        raise ValueError("fixture snapshot lock does not match deterministic fixtures")
    parameter_path = repository / str(value["backend_parameter_lock_path"])
    if file_sha256(parameter_path) != value["backend_parameter_lock_sha256"]:
        raise ValueError("fixture backend parameter lock SHA mismatch")
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    if parameters.get("formal_seed_values_included") is not False:
        raise ValueError("fixture backend parameter lock contains formal seed values")
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan_value != fixture_plan_payload():
        raise ValueError("fixture plan does not match deterministic fixtures")
    return value, build_fixture_snapshots()


def materialize_fixture_cache(directory: str | Path, snapshots: tuple[FixtureSnapshot, ...]) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    import json

    for item in snapshots:
        token = item.snapshot_id.rsplit("/", 1)[-1]
        snapshot_dir = destination / token
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        np.save(snapshot_dir / "source_points.npy", item.source, allow_pickle=False)
        np.save(snapshot_dir / "target_points.npy", item.target, allow_pickle=False)
        np.save(snapshot_dir / "reference_pose.npy", item.reference, allow_pickle=False)
        metadata = {
            "checksums": dict(item.checksums),
            "condition": item.condition,
            "fixture_only": True,
            "formal_phase_a": False,
            "scene_variant": item.scene_variant,
            "snapshot_id": item.snapshot_id,
        }
        (snapshot_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "FIXTURE_LOCK_RELATIVE",
    "FIXTURE_PARAMETER_LOCK_RELATIVE",
    "FIXTURE_PLAN_RELATIVE",
    "FORMAL_CACHE_RELATIVE",
    "FixtureSnapshot",
    "build_fixture_snapshots",
    "fixture_lock_payload",
    "fixture_plan_payload",
    "materialize_fixture_cache",
    "validate_fixture_lock",
]
