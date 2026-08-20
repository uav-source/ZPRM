from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pytest

from experiments.mid360_formal_batch1 import zero_perturbation_v1_1_r1_verify as independent

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_lock import (
    DEFAULT_BINDING_PATHS,
    LOCK_FILENAME,
)
from experiments.mid360_formal_batch1.zero_perturbation_v1_1_r1_verify import (
    DEFAULT_AUTHORIZATION_PATH,
    R1IndependentVerificationError,
    verify_release_checksums,
    verify_r1_lock,
)
from tests.mid360_formal_batch1.zero_perturbation_v1_1_r1_fixture import (
    COMMIT,
    build_valid_r1_lock,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _rewrite(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bound(root: Path, name: str) -> Path:
    return root / DEFAULT_BINDING_PATHS[name]


def _verify(root: Path, lock_dir: Path, runtime: Path) -> dict[str, object]:
    return verify_r1_lock(
        root, lock_dir, expected_execution_code_commit=COMMIT,
        runtime_root=runtime, remeasure_environment=False,
        remeasure_execution_commit=False,
    )


def _dataset_bindings() -> dict[str, dict[str, str]]:
    return {
        name: {"repository_relative_path": relative}
        for name, relative in DEFAULT_BINDING_PATHS.items()
    }


def test_real_final_station_registry_mixed_attempt_schemas_pass() -> None:
    snapshots, targets = independent._verify_final_dataset(
        REPOSITORY, _dataset_bindings()
    )
    assert len(snapshots) == 180
    assert len(targets) == 18
    registry = json.loads(
        (REPOSITORY / DEFAULT_BINDING_PATHS["final_station_registry"]).read_text()
    )
    w02 = [row for row in registry["stations"] if row["scene_id"] == "FMB1_W02"]
    assert len(w02) == 3
    assert all(row["attempt"] == 2 for row in w02)
    assert all("acquisition_status" not in row for row in w02)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt", 1),
        ("station_acquisition_status", "ACQUISITION_FAIL"),
        ("attempt_status", "INVALID_ACQUISITION"),
        ("map_bag_status", "FAIL"),
        ("query_bag_status", "FAIL"),
        ("pair_audit.FORMAL_PAIR_VALID", False),
        ("acquisition_status", "ACQUISITION_PASS"),
    ],
)
def test_w02_attempt2_station_schema_tamper_fails_closed(
    tmp_path: Path, field: str, value: object,
) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    registry = _bound(root, "final_station_registry")

    def tamper(payload: dict[str, object]) -> None:
        stations = payload["stations"]
        assert isinstance(stations, list)
        row = next(
            item for item in stations
            if isinstance(item, dict) and item.get("scene_id") == "FMB1_W02"
        )
        if field == "pair_audit.FORMAL_PAIR_VALID":
            pair = row["pair_audit"]
            assert isinstance(pair, dict)
            pair["FORMAL_PAIR_VALID"] = value
        else:
            row[field] = value

    _rewrite(registry, tamper)
    with pytest.raises(
        R1IndependentVerificationError,
        match="W02 attempt2 station acquisition record differs",
    ):
        independent._verify_final_dataset(root, _dataset_bindings())


def test_independent_verifier_recomputes_valid_lock_without_backend(tmp_path: Path) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    report = _verify(root, lock_dir, runtime)
    assert report["pass"] is True
    assert report["planned_total_trials"] == 360
    assert report["FORMAL_REGISTRATION_AUTHORIZED"] is False
    assert report["actual_formal_trials"] == 0
    assert report["registration_backend_imports_or_calls"] == 0


def _tamper_w04(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "final_dataset_pointer"), lambda d: d.__setitem__("W04_INCLUDED_IN_FINAL_SET", True))


def _tamper_old_w02(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "acquisition_attempt_lineage"), lambda d: d.__setitem__("invalid_attempt_snapshot_count_in_final", 1))


def _delete_lineage(root: Path, lock: Path, runtime: Path) -> None:
    _bound(root, "acquisition_attempt_lineage").unlink()


def _proposal_status(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "active_amendment"), lambda d: d.__setitem__("status", "PROPOSED"))


def _activation_after_icp(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "amendment_activation_record"), lambda d: d.__setitem__("ACTIVATED_BEFORE_ANY_FORMAL_ICP", False))


def _plan_359(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"].pop())  # type: ignore[index,union-attr]


def _backend_imbalance(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0].__setitem__("backend", "PCL_POINT_TO_PLANE"))  # type: ignore[index,union-attr]


def _nonidentity(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0]["T0"][0].__setitem__(3, 0.01))  # type: ignore[index,union-attr]


def _capture_radius(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0].__setitem__("translation_perturbation_m", 0.1))  # type: ignore[index,union-attr]


def _source_sha(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0].__setitem__("source_sha256", "0" * 64))  # type: ignore[index,union-attr]


def _target_sha(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0].__setitem__("target_sha256", "0" * 64))  # type: ignore[index,union-attr]


def _geometry_class(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "trial_plan_json"), lambda d: d["rows"][0].__setitem__("final_geometry_class", "WEAK"))  # type: ignore[index,union-attr]


def _backend_contract(root: Path, lock: Path, runtime: Path) -> None:
    _bound(root, "backend_parameter_contract").write_bytes(
        _bound(root, "backend_parameter_contract").read_bytes() + b"\n"
    )


def _pcl_binary(root: Path, lock: Path, runtime: Path) -> None:
    path = _bound(root, "pcl_executable")
    path.chmod(path.stat().st_mode & ~0o111)


def _physical_gt(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "analysis_contract"), lambda d: d["physical_reference"].__setitem__("physical_reference_semantics", "SUBMILLIMETER_GT"))  # type: ignore[index,union-attr]


def _authorization_true(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(lock / LOCK_FILENAME, lambda d: d.__setitem__("FORMAL_REGISTRATION_AUTHORIZED", True))


def _real_result(root: Path, lock: Path, runtime: Path) -> None:
    path = runtime / "trial_results/forbidden/attempt-0001.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")


def _execution_commit(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(lock / LOCK_FILENAME, lambda d: d.__setitem__("execution_code_commit", "b" * 40))


def _fingerprint(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(lock / "lock_fingerprint.json", lambda d: d.__setitem__("lock_fingerprint", "0" * 64))


def _snapshot_independent(root: Path, lock: Path, runtime: Path) -> None:
    _rewrite(_bound(root, "analysis_contract"), lambda d: d["experimental_units"].__setitem__("snapshots_are_independent_scenes", True))  # type: ignore[index,union-attr]


TAMPERS = [
    _tamper_w04, _tamper_old_w02, _delete_lineage, _proposal_status,
    _activation_after_icp, _plan_359, _backend_imbalance, _nonidentity,
    _capture_radius, _source_sha, _target_sha, _geometry_class,
    _backend_contract, _pcl_binary, _physical_gt, _authorization_true,
    _real_result, _execution_commit, _fingerprint, _snapshot_independent,
]


@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda value: value.__name__)
def test_twenty_required_tamper_classes_fail_closed(
    tmp_path: Path, tamper: Callable[[Path, Path, Path], None],
) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    tamper(root, lock_dir, runtime)
    with pytest.raises((R1IndependentVerificationError, FileNotFoundError, OSError)):
        _verify(root, lock_dir, runtime)


def test_core_checksum_path_traversal_is_rejected(tmp_path: Path) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    sums = lock_dir / "LOCK_CORE_SHA256SUMS"
    sums.write_text(sums.read_text() + f"{'0'*64}  ../escape\n")
    with pytest.raises(R1IndependentVerificationError, match="unsafe"):
        _verify(root, lock_dir, runtime)


@pytest.mark.parametrize(
    "relative",
    ("run_contract.json", "inflight/trial-0001.started.json", "run_manifest.json"),
)
def test_independent_verifier_rejects_any_execution_lifecycle_file(
    tmp_path: Path, relative: str,
) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    marker = runtime / relative
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n")
    with pytest.raises(R1IndependentVerificationError, match="lifecycle"):
        _verify(root, lock_dir, runtime)


def test_independent_verifier_rejects_existing_authorization(tmp_path: Path) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    (root / DEFAULT_AUTHORIZATION_PATH).write_text("{}\n")
    with pytest.raises(R1IndependentVerificationError, match="authorization"):
        _verify(root, lock_dir, runtime)


def test_final_release_checksum_verifier_has_exact_set_and_no_self_reference(
    tmp_path: Path,
) -> None:
    _, lock_dir, _ = build_valid_r1_lock(tmp_path)
    report = verify_release_checksums(lock_dir)
    assert report["pass"] is True
    assert report["set_equality_verified"] is True
    assert report["covered_regular_file_count"] > 0


@pytest.mark.parametrize("tamper", ["extra", "missing", "duplicate", "traversal", "symlink"])
def test_final_release_checksum_verifier_fails_closed(
    tmp_path: Path, tamper: str,
) -> None:
    _, lock_dir, _ = build_valid_r1_lock(tmp_path)
    sums = lock_dir / "SHA256SUMS"
    lines = sums.read_text().splitlines()
    if tamper == "extra":
        (lock_dir / "unlisted.json").write_text("{}\n")
    elif tamper == "missing":
        sums.write_text("\n".join(lines[:-1]) + "\n")
    elif tamper == "duplicate":
        sums.write_text("\n".join(lines + [lines[0]]) + "\n")
    elif tamper == "traversal":
        sums.write_text(sums.read_text() + f"{'0'*64}  ../escape\n")
    else:
        os.symlink(lock_dir / LOCK_FILENAME, lock_dir / "linked-lock.json")
    with pytest.raises(R1IndependentVerificationError):
        verify_release_checksums(lock_dir)


@pytest.mark.parametrize(
    "binding_name", [
        "execution_runner_cli", "execution_metrics", "execution_types",
        "execution_experiments_package_init",
        "execution_mid360_formal_batch1_package_init",
        "execution_phase_a_harness_package_init",
    ],
)
def test_transitive_execution_code_tamper_fails_independent_lock_verification(
    tmp_path: Path, binding_name: str,
) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    path = _bound(root, binding_name)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(R1IndependentVerificationError, match="bound file changed"):
        _verify(root, lock_dir, runtime)


@pytest.mark.parametrize(
    "binding_name", [
        "execution_experiments_package_init",
        "execution_mid360_formal_batch1_package_init",
        "execution_phase_a_harness_package_init",
    ],
)
def test_independent_ast_rejects_eager_package_initializer(
    tmp_path: Path, binding_name: str,
) -> None:
    root, _, _ = build_valid_r1_lock(tmp_path)
    path = _bound(root, binding_name)
    path.write_text(path.read_text() + "\nimport importlib\nimportlib.import_module('math')\n")
    with pytest.raises(R1IndependentVerificationError, match="initializer"):
        independent._verify_execution_boundary(root)


@pytest.mark.parametrize(
    ("binding_name", "field"),
    [
        ("proposal_correction_record", "actual_formal_trials"),
        ("analysis_missingness_clarification", "clarification_at_formal_trial_count"),
        ("analysis_missingness_clarification_transition", "actual_formal_trials"),
        ("protocol_c1_missingness_independent_verification", "actual_formal_trials"),
        ("activation_review", "review_status"),
    ],
)
def test_provenance_and_c1_tamper_fail_independent_verification(
    tmp_path: Path, binding_name: str, field: str,
) -> None:
    root, lock_dir, runtime = build_valid_r1_lock(tmp_path)
    _rewrite(_bound(root, binding_name), lambda payload: payload.__setitem__(field, 1))
    with pytest.raises(R1IndependentVerificationError):
        _verify(root, lock_dir, runtime)
