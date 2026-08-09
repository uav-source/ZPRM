from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from phase_a_harness.contracts import file_sha256
from phase_a_harness.fixture_publication import (
    audit_and_publish_existing_fixture_results,
)
from phase_a_harness.fixture_publication_artifact_verifier import (
    verify_fixture_publication_artifact,
)
from phase_a_harness.phase_a_trial_result_schema import canonical_json_bytes
from phase_a_harness.synthetic_confirmatory_analysis import (
    BACKENDS,
    CONDITIONS,
    SCENES,
    analyze_synthetic_confirmatory_records,
    load_synthetic_confirmatory_raw,
)
from phase_a_harness.synthetic_confirmatory_artifact_verifier import (
    FORMAL_REQUIRED_FILES,
    PRERUN_REQUIRED_GATE_NAMES,
    PRERUN_ROOT_FILES,
    PRERUN_ZERO_EXECUTION_COUNTER_NAMES,
    verify_synthetic_confirmatory_artifact,
    verify_synthetic_confirmatory_prerun_artifact,
)
from phase_a_harness.synthetic_confirmatory_independent_verifier import (
    compare_primary_and_independent,
    independently_recompute_synthetic_confirmatory,
)
from phase_a_harness.synthetic_confirmatory_publisher import (
    _publish_into,
    publish_synthetic_confirmatory,
)


GEOMETRIES = (11, 12, 13, 14, 15)
MEASUREMENTS = (21, 22, 23)
ROOT = Path(__file__).resolve().parents[1]


def _contract(root: Path) -> dict[str, object]:
    return json.loads(
        (root / "protocols/synthetic_confirmatory_gate_contract.json").read_text(
            encoding="utf-8"
        )
    )


def _matrix() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scale = {
        "GEOMETRY_RICH_ROOM": 0.0012,
        "END_FACE_TRANSITION_PRESENT": 0.003,
        "END_FACE_TRANSITION_WEAK": 0.006,
        "PARALLEL_WALLS": 0.012,
        "END_FACE_TRANSITION_ABSENT": 0.014,
        "REPEATED_STRUCTURE": 0.016,
        "LONG_CORRIDOR": 0.020,
    }
    trials: list[dict[str, object]] = []
    common: list[dict[str, object]] = []
    for backend_index, backend in enumerate(BACKENDS):
        for scene in SCENES:
            for geometry_index, geometry in enumerate(GEOMETRIES):
                specifications = [("IDEAL_MATCHED", None, 0)]
                specifications += [("INDEPENDENT_NOISE_FREE", None, 0)]
                specifications += [
                    ("FULL_NOISE", measurement, repeat)
                    for measurement in MEASUREMENTS for repeat in range(5)
                ]
                for condition, measurement, repeat in specifications:
                    snapshot_id = (
                        f"artificial/{scene}/{condition}/{geometry}/"
                        f"{measurement}/{repeat}"
                    )
                    trial_id = f"{snapshot_id}/{backend}"
                    if condition == "IDEAL_MATCHED":
                        error = 1.0e-6 * (1.0 + 0.01 * geometry_index)
                    else:
                        condition_factor = (
                            0.9 if condition == "INDEPENDENT_NOISE_FREE" else 1.0
                        )
                        replicate_index = (
                            0 if measurement is None
                            else MEASUREMENTS.index(measurement) * 5 + repeat
                        )
                        error = (
                            scale[scene]
                            * condition_factor
                            * (1.0 + 0.03 * geometry_index)
                            * (1.0 + 0.004 * replicate_index)
                            * (1.0 + 0.02 * backend_index)
                        )
                    row = {
                        "backend_schema_name": backend,
                        "condition": condition,
                        "finite_output": True,
                        "geometry_seed": geometry,
                        "measurement_seed": measurement,
                        "planned_snapshot_id": snapshot_id,
                        "planned_trial_id": trial_id,
                        "repeat_index": repeat,
                        "rotation_error_rad": error / 100.0,
                        "scene_variant": scene,
                        "solver_failure": False,
                        "source_checksum": hashlib.sha256(
                            f"source/{snapshot_id}".encode()
                        ).hexdigest(),
                        "target_checksum": hashlib.sha256(
                            f"target/{snapshot_id}".encode()
                        ).hexdigest(),
                        "translation_error_m": error,
                        "translation_vector": [error, 0.0, 0.0],
                    }
                    trials.append(row)
                    if condition != "IDEAL_MATCHED":
                        common.append({
                            "backend_schema_name": backend,
                            "common_association_valid": True,
                            "correspondence_turnover": error,
                            "planned_trial_id": trial_id,
                            "target_feature": __import__("math").log10(error + 1e-9),
                        })
    return trials, common


def _models(*, good_b: bool = True) -> dict[str, object]:
    rows = []
    for backend in BACKENDS:
        rows.extend([
            {
                "backend_schema_name": backend,
                "model": "A",
                "feature_names": ["target_feature"],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "coefficient": [0.0],
                "intercept": -1.0,
            },
            {
                "backend_schema_name": backend,
                "model": "B",
                "feature_names": ["target_feature"],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "coefficient": [1.0 if good_b else 0.0],
                "intercept": 0.0 if good_b else -1.0,
            },
        ])
    return {"models": rows}


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _artificial_snapshot(
    root: Path,
    *,
    source_dtype: str = "<f4",
    nonfinite_source: bool = False,
    aggregate_override: str | None = None,
):
    cache = root / "cache"
    snapshot_id = "artificial-snapshot"
    directory = cache / snapshot_id
    directory.mkdir(parents=True)
    plan = {
        "planned_snapshot_id": snapshot_id,
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "condition": "IDEAL_MATCHED",
        "geometry_seed": GEOMETRIES[0],
        "measurement_seed": None,
        "repeat_index": 0,
    }
    target = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype="<f4",
    )
    source = target.astype(source_dtype)
    if nonfinite_source:
        source[0, 0] = np.nan
    reference = np.eye(4, dtype="<f8")
    for name, array in (
        ("source_points.npy", source),
        ("target_points.npy", target),
        ("reference_pose.npy", reference),
    ):
        with (directory / name).open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
    raw = {
        "source_checksum": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
        "target_checksum": hashlib.sha256(target.tobytes(order="C")).hexdigest(),
        "reference_pose_checksum": hashlib.sha256(
            reference.tobytes(order="C")
        ).hexdigest(),
    }
    snapshot_checksum = _canonical_sha({"snapshot_id": snapshot_id, **raw})
    metadata = {
        "array_file_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "reference_pose.npy", "source_points.npy", "target_points.npy"
            )
        },
        "condition": "IDEAL_MATCHED",
        "confirmatory_rng_instantiation_count": 0,
        "development_protocol_sha256": "a" * 64,
        "dropout_parameters": {
            "map_dropout_fraction": 0.0,
            "scan_dropout_fraction": 0.0,
        },
        "generator_sha256": "b" * 64,
        "geometry_seed": GEOMETRIES[0],
        "independent_sampling": False,
        "initial_pose": "reference_pose_exact",
        "measurement_seed": None,
        "noise_parameters": {
            "map_noise_sigma_m": 0.0,
            "scan_noise_sigma_m": 0.0,
        },
        "planned_snapshot_id": snapshot_id,
        "reference_pose_checksum": raw["reference_pose_checksum"],
        "repeat_index": 0,
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "snapshot_builder_contract_version": (
            "synthetic_confirmatory_snapshot_builder_v1"
        ),
        "snapshot_checksum": aggregate_override or snapshot_checksum,
        "snapshot_id": snapshot_id,
        "source_checksum": raw["source_checksum"],
        "source_is_target_subset": True,
        "source_point_count": len(source),
        "target_checksum": raw["target_checksum"],
        "target_point_count": len(target),
    }
    metadata["metadata_payload_sha256"] = _canonical_sha(metadata)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    file_sha = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "metadata.json", "source_points.npy", "target_points.npy",
            "reference_pose.npy",
        )
    }
    entry = {
        "condition": "IDEAL_MATCHED",
        "confirmatory_rng_instantiation_count": 0,
        "file_sha256": file_sha,
        "geometry_seed": GEOMETRIES[0],
        "measurement_seed": None,
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_pose_checksum": raw["reference_pose_checksum"],
        "repeat_index": 0,
        "scene_variant": "GEOMETRY_RICH_ROOM",
        "snapshot_checksum": aggregate_override or snapshot_checksum,
        "snapshot_id": snapshot_id,
        "source_checksum": raw["source_checksum"],
        "target_checksum": raw["target_checksum"],
    }
    core = {
        "condition_snapshot_counts": {"IDEAL_MATCHED": 1},
        "confirmatory_rng_instantiation_count": 0,
        "planned_snapshot_count": 1,
        "schema_version": "synthetic_confirmatory_snapshot_lock_v1",
        "snapshot_builder_contract_version": (
            "synthetic_confirmatory_snapshot_builder_v1"
        ),
        "snapshots": [entry],
    }
    lock = {**core, "snapshot_lock_payload_sha256": _canonical_sha(core)}
    lock_path = root / "snapshot_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache, plan, entry, lock_path


def _artificial_formal_manifest(root: Path) -> Path:
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _INDEPENDENT_MANIFEST_BINDINGS,
    )

    bound = {}
    for name, relative in _INDEPENDENT_MANIFEST_BINDINGS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artificial binding {name}\n", encoding="utf-8")
        bound[name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "backend_count": 2,
        "bootstrap_seed": 1083684578,
        "bound_files": bound,
        "formal_execution_authorized": True,
        "formal_output_dir": "results/synthetic_confirmatory_v1",
        "formal_run_id": "synthetic-confirmatory-v1",
        "formal_workers": 2,
        "manifest_version": "1",
        "native_trial_count": 0,
        "open3d_parameter_sha256": (
            "94a2d1e991658b7088be43ad1a82f9b1d783264736c3beb0217d6dd1c7a26413"
        ),
        "open3d_version": "artificial",
        "pcl_parameter_sha256": (
            "16b3d124f466f33c41a1ecfb27a103a570c3ad2055db9c87ae92c74dbba64abd"
        ),
        "pcl_version": "artificial",
        "planned_snapshot_count": 595,
        "planned_trial_count": 1190,
        "raw_result_manifest_schema": (
            "synthetic_confirmatory_raw_result_manifest_v1"
        ),
        "scientific_survival_commit": (
            "ffc15334f4ded25fdba5e709b45657dbad481dfc"
        ),
        "scientific_survival_tag": (
            "archive/zero-perturbation-scientific-survival-audit-v1"
        ),
        "snapshot_cache_root": "data/synthetic_confirmatory_v1_snapshots",
    }
    for name in (
        "scientific_protocol", "scientific_protocol_document", "gate_contract",
        "planned_snapshots", "planned_trials", "seed_provenance_audit",
        "frozen_model",
    ):
        payload[f"{name}_path"] = bound[name]["path"]
        payload[f"{name}_sha256"] = bound[name]["sha256"]
    manifest = {**payload, "manifest_payload_sha256": _canonical_sha(payload)}
    path = root / "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture(scope="module")
def evidence() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return _matrix()


@pytest.fixture(scope="module")
def gate_contract() -> dict[str, object]:
    return _contract(Path(__file__).resolve().parents[1])


def _primary(evidence, gate_contract, *, models=None):
    trials, common = evidence
    return analyze_synthetic_confirmatory_records(
        trials=trials,
        common_records=common,
        model_lock=_models() if models is None else models,
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )


def test_artificial_matrix_has_exact_frozen_cardinality(evidence):
    trials, common = evidence
    assert len(trials) == 1190
    assert len(common) == 1120
    assert {row["condition"] for row in trials} == set(CONDITIONS)


def test_h1_through_h6_pass_on_seed_free_artificial_matrix(evidence, gate_contract):
    report = _primary(evidence, gate_contract)
    assert all(report["gate_summary"].values())
    assert report["final_decision"]["SYNTHETIC_CONFIRMATORY_COMPLETE"] is True
    assert report["final_decision"]["SYNTHETIC_CONFIRMATORY_PASS"] is True


def test_independent_recomputation_has_zero_difference(evidence, gate_contract):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    comparison = compare_primary_and_independent(primary, independent)
    assert comparison["section_difference_count"] == 0
    assert comparison["leaf_difference_count"] == 0


def test_independent_module_does_not_import_primary_aggregation():
    import phase_a_harness.synthetic_confirmatory_independent_verifier as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_modules = {
        "phase_a_harness.synthetic_confirmatory_analysis",
        "phase_a_harness.common_association_analysis",
        "phase_a_harness.synthetic_confirmatory_snapshot_builder",
        "phase_a_harness.confirmatory_protocol",
        "phase_a_harness.synthetic_confirmatory_manifest",
        "synthetic_confirmatory_analysis",
        "common_association_analysis",
        "synthetic_confirmatory_snapshot_builder",
        "confirmatory_protocol",
        "synthetic_confirmatory_manifest",
    }
    assert not any(
        module_name.lstrip(".") in forbidden_modules
        or any(module_name.endswith(f".{name}") for name in forbidden_modules)
        for module_name in imported_modules
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "read_synthetic_confirmatory_snapshot" not in called_names
    assert "verify_synthetic_confirmatory_manifest" not in called_names


def test_independent_manifest_validator_authenticates_payload_and_every_binding(
    tmp_path: Path,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_validate_formal_manifest,
    )

    manifest_path = _artificial_formal_manifest(tmp_path)
    repository, manifest = _independent_validate_formal_manifest(manifest_path)
    assert repository == tmp_path.resolve()
    assert manifest["formal_execution_authorized"] is True
    bound_path = tmp_path / manifest["bound_files"]["analysis"]["path"]
    bound_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        _independent_validate_formal_manifest(manifest_path)


def test_independent_snapshot_reader_and_lock_accept_valid_artificial_asset(
    tmp_path: Path,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_read_snapshot,
        _independent_validate_snapshot_lock,
    )

    cache, plan, _entry, lock_path = _artificial_snapshot(tmp_path)
    _lock, entries = _independent_validate_snapshot_lock(
        lock_path, [plan], expected_snapshot_count=1
    )
    snapshot = _independent_read_snapshot(
        cache, plan, expected_lock_entry=entries[plan["planned_snapshot_id"]]
    )
    assert snapshot["source"].dtype == np.dtype("<f4")
    assert snapshot["target"].dtype == np.dtype("<f4")
    assert snapshot["reference"].dtype == np.dtype("<f8")


def test_independent_snapshot_reader_rejects_extra_file(tmp_path: Path):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_read_snapshot,
    )

    cache, plan, entry, _lock_path = _artificial_snapshot(tmp_path)
    (cache / plan["planned_snapshot_id"] / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="four-file inventory"):
        _independent_read_snapshot(cache, plan, expected_lock_entry=entry)


def test_independent_snapshot_reader_rejects_duplicate_metadata_key(
    tmp_path: Path,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_read_snapshot,
    )

    cache, plan, entry, _lock_path = _artificial_snapshot(tmp_path)
    path = cache / plan["planned_snapshot_id"] / "metadata.json"
    original = path.read_text(encoding="utf-8").lstrip()
    path.write_text(
        '{"condition":"IDEAL_MATCHED",' + original[1:], encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _independent_read_snapshot(cache, plan, expected_lock_entry=entry)


@pytest.mark.parametrize(
    ("source_dtype", "nonfinite", "error"),
    [("<f8", False, "dtype/shape/finite"), ("<f4", True, "dtype/shape/finite")],
)
def test_independent_snapshot_reader_rejects_semantic_array_tamper_with_fresh_sha(
    tmp_path: Path,
    source_dtype: str,
    nonfinite: bool,
    error: str,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_read_snapshot,
    )

    cache, plan, entry, _lock_path = _artificial_snapshot(
        tmp_path, source_dtype=source_dtype, nonfinite_source=nonfinite
    )
    with pytest.raises(ValueError, match=error):
        _independent_read_snapshot(cache, plan, expected_lock_entry=entry)


def test_independent_snapshot_reader_rejects_aggregate_and_lock_tamper(
    tmp_path: Path,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_read_snapshot,
        _independent_validate_snapshot_lock,
    )

    cache, plan, entry, lock_path = _artificial_snapshot(
        tmp_path, aggregate_override="c" * 64
    )
    with pytest.raises(ValueError, match="raw/aggregate checksum"):
        _independent_read_snapshot(cache, plan, expected_lock_entry=entry)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["snapshots"][0]["source_checksum"] = "d" * 64
    unsigned = {
        key: value for key, value in lock.items()
        if key != "snapshot_lock_payload_sha256"
    }
    lock["snapshot_lock_payload_sha256"] = _canonical_sha(unsigned)
    lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
    _lock, entries = _independent_validate_snapshot_lock(
        lock_path, [plan], expected_snapshot_count=1
    )
    with pytest.raises(ValueError, match="raw/aggregate checksum"):
        _independent_read_snapshot(
            cache, plan,
            expected_lock_entry=entries[plan["planned_snapshot_id"]],
        )


def test_independent_snapshot_lock_rejects_duplicate_key_or_inventory_tamper(
    tmp_path: Path,
):
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_validate_snapshot_lock,
    )

    _cache, plan, _entry, lock_path = _artificial_snapshot(tmp_path)
    original = lock_path.read_text(encoding="utf-8").lstrip()
    lock_path.write_text(
        '{"planned_snapshot_count":1,' + original[1:], encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _independent_validate_snapshot_lock(
            lock_path, [plan], expected_snapshot_count=1
        )


def test_independent_common_recomputation_is_exact_on_artificial_arrays():
    from phase_a_harness.common_association_analysis import (
        prepare_common_association_context,
        safe_analyze_estimated_transform,
    )
    from phase_a_harness.synthetic_confirmatory_independent_verifier import (
        _independent_prepare_common,
        _independent_safe_common_record,
    )

    axis = np.linspace(-1.0, 1.0, 8)
    points = []
    for first in axis:
        for second in axis:
            points.extend(
                ([first, second, 0.0], [first, 0.2, second], [0.3, first, second])
            )
    target = np.asarray(points, dtype=np.float32)
    source = target.copy()
    reference = np.eye(4, dtype=np.float64)
    estimated = np.eye(4, dtype=np.float64)
    estimated[:3, 3] = [0.01, -0.002, 0.003]
    identifiers = {
        "planned_trial_id": "artificial-common/open3d",
        "backend_schema_name": BACKENDS[0],
        "scene_variant": SCENES[0],
        "condition": "FULL_NOISE",
        "geometry_seed": GEOMETRIES[0],
        "measurement_seed": MEASUREMENTS[0],
        "repeat_index": 0,
    }
    primary = safe_analyze_estimated_transform(
        prepare_common_association_context(
            source, target, reference, snapshot_id="artificial-common"
        ),
        estimated,
        identifiers=identifiers,
    )
    independent = _independent_safe_common_record(
        _independent_prepare_common(
            source, target, reference, snapshot_id="artificial-common"
        ),
        estimated,
        identifiers=identifiers,
    )
    assert independent == primary


def test_h1_failure_is_not_reclassified(evidence, gate_contract):
    trials, common = copy.deepcopy(evidence)
    for row in trials:
        if row["condition"] == "IDEAL_MATCHED":
            row["translation_error_m"] = 0.01
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H1_IDEAL_CONTROL_PASS"] is False
    assert report["final_decision"]["SYNTHETIC_CONFIRMATORY_PASS"] is False


def test_h2_geometry_block_gate_fails_closed(evidence, gate_contract):
    trials, common = copy.deepcopy(evidence)
    for row in trials:
        if row["scene_variant"] == "LONG_CORRIDOR":
            row["translation_error_m"] = 0.001
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H2_LONG_CORRIDOR_SCENE_EFFECT_PASS"] is False


def test_h4_turnover_gate_fails_on_constant_signal(evidence, gate_contract):
    trials, common = copy.deepcopy(evidence)
    for row in common:
        row["correspondence_turnover"] = 0.5
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H4_REASSOCIATION_MECHANISM_PASS"] is False


def test_h4_uses_condition_cells_and_log_error_with_exact_independent_match(
    evidence, gate_contract
):
    from scipy.stats import spearmanr

    trials, common = copy.deepcopy(evidence)
    trial_by_id = {row["planned_trial_id"]: row for row in trials}
    for metric in common:
        trial = trial_by_id[metric["planned_trial_id"]]
        log_error = math.log10(float(trial["translation_error_m"]) + 1.0e-9)
        condition_offset = (
            0.7 if trial["condition"] == "INDEPENDENT_NOISE_FREE" else 0.0
        )
        metric["correspondence_turnover"] = 0.05 * (log_error + 4.0) + condition_offset
    primary = _primary((trials, common), gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    comparison = compare_primary_and_independent(primary, independent)
    assert comparison["leaf_difference_count"] == 0
    assert comparison["maximum_absolute_numeric_difference"] == 0.0
    metric_by_id = {row["planned_trial_id"]: row for row in common}
    for h4_row in primary["h4_reassociation"]:
        backend = h4_row["backend_schema_name"]
        selected = [
            {**row, **metric_by_id[row["planned_trial_id"]]}
            for row in trials
            if row["backend_schema_name"] == backend
            and row["condition"] != "IDEAL_MATCHED"
        ]
        correct_x, correct_y = [], []
        old_x, old_y = [], []
        condition_cells = {}
        scene_cells = {}
        for row in selected:
            condition_cells.setdefault(
                (row["scene_variant"], row["condition"]), []
            ).append(row)
            scene_cells.setdefault(row["scene_variant"], []).append(row)
        for rows in condition_cells.values():
            turnover = np.asarray(
                [float(row["correspondence_turnover"]) for row in rows]
            )
            log_error = np.asarray([
                math.log10(float(row["translation_error_m"]) + 1.0e-9)
                for row in rows
            ])
            correct_x.extend(turnover - np.median(turnover))
            correct_y.extend(log_error - np.median(log_error))
        for rows in scene_cells.values():
            turnover = np.asarray(
                [float(row["correspondence_turnover"]) for row in rows]
            )
            raw_error = np.asarray(
                [float(row["translation_error_m"]) for row in rows]
            )
            old_x.extend(turnover - np.median(turnover))
            old_y.extend(raw_error - np.median(raw_error))
        expected = float(spearmanr(correct_x, correct_y).statistic)
        old_value = float(spearmanr(old_x, old_y).statistic)
        assert h4_row["error_transform"] == "log10(translation_error_m+1e-9)"
        assert h4_row["centering_cell"] == ["scene_variant", "condition"]
        assert h4_row["scene_centered_spearman_rho"] == pytest.approx(
            expected, abs=1.0e-15
        )
        assert abs(expected - old_value) > 1.0e-3


def test_h5_uses_frozen_predictions_without_refit(evidence, gate_contract):
    report = _primary(evidence, gate_contract, models=_models(good_b=False))
    assert report["gate_summary"]["H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS"] is False


def test_h5_rejects_any_candidate_with_missing_or_nonfinite_feature(
    evidence, gate_contract
):
    trials, common = copy.deepcopy(evidence)
    for backend in BACKENDS:
        selected = next(
            row for row in common if row["backend_schema_name"] == backend
        )
        selected.pop("target_feature")
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS"] is False
    assert [row["invalid_count"] for row in report["h5_frozen_models"]] == [1, 1]
    assert all(
        row["candidate_count"] == row["evaluated_count"] + row["invalid_count"]
        for row in report["h5_frozen_models"]
    )
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    assert independent["final_decision"][
        "H5_FROZEN_MODEL_B_INCREMENTAL_VALUE_PASS"
    ] is False
    assert compare_primary_and_independent(report, independent)[
        "leaf_difference_count"
    ] == 0


def test_h6_systematic_offset_gate_fails_on_cancelling_vectors(evidence, gate_contract):
    trials, common = copy.deepcopy(evidence)
    for row in trials:
        if row["condition"] == "FULL_NOISE" and row["scene_variant"] == "LONG_CORRIDOR":
            sign = -1.0 if int(row["repeat_index"]) % 2 else 1.0
            error = float(row["translation_error_m"])
            row["translation_vector"] = [sign * error, 0.0, 0.0]
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS"] is False


def test_h6_allows_three_failed_trials_when_twelve_effective_replicates_remain(
    evidence, gate_contract
):
    trials, common = copy.deepcopy(evidence)
    failed_ids = set()
    for backend in BACKENDS:
        for geometry in GEOMETRIES:
            candidates = sorted(
                (
                    row for row in trials
                    if row["backend_schema_name"] == backend
                    and row["condition"] == "FULL_NOISE"
                    and row["scene_variant"] == "LONG_CORRIDOR"
                    and row["geometry_seed"] == geometry
                ),
                key=lambda row: (row["measurement_seed"], row["repeat_index"]),
            )
            for row in candidates[:3]:
                row["solver_failure"] = True
                row["finite_output"] = False
                failed_ids.add(row["planned_trial_id"])
    common = [row for row in common if row["planned_trial_id"] not in failed_ids]
    report = _primary((trials, common), gate_contract)
    assert report["gate_summary"]["H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS"] is True
    assert all(row["successful_count"] == 12 for row in report["h6_systematic_groups"])
    assert all(row["effective_replicate_count"] == 12 for row in report["h6_systematic_groups"])
    assert all(row["group_gate_pass"] for row in report["h6_systematic_groups"])
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    assert independent["final_decision"][
        "H6_FULL_NOISE_SYSTEMATIC_OFFSET_PASS"
    ] is True
    assert compare_primary_and_independent(report, independent)[
        "leaf_difference_count"
    ] == 0


def test_any_nonzero_numeric_difference_is_reported(evidence, gate_contract):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    independent = copy.deepcopy(independent)
    independent["verification_projection"]["h3_cross_backend_ranking"][0][
        "spearman_rho"
    ] -= 5.0e-13
    comparison = compare_primary_and_independent(primary, independent)
    assert comparison["leaf_difference_count"] == 1
    assert comparison["maximum_absolute_numeric_difference"] > 0.0


def test_missing_raw_results_refuse_before_publication_output(tmp_path: Path):
    artifact = tmp_path / "artifact"
    with pytest.raises(FileNotFoundError):
        publish_synthetic_confirmatory(
            manifest_path=tmp_path / "missing-manifest.json",
            run_dir=tmp_path / "missing-results",
            primary={},
            independent={},
            artifact_dir=artifact,
        )
    assert not artifact.exists()
    with pytest.raises(FileNotFoundError):
        load_synthetic_confirmatory_raw(
            manifest_path=tmp_path / "missing-manifest.json",
            run_dir=tmp_path / "missing-results",
        )


def _prerun_decision() -> dict[str, object]:
    return {
        **{name: True for name in PRERUN_REQUIRED_GATE_NAMES},
        **{name: 0 for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES},
        "required_gate_names": list(PRERUN_REQUIRED_GATE_NAMES),
        "zero_execution_counter_names": list(PRERUN_ZERO_EXECUTION_COUNTER_NAMES),
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS": True,
        "CONFIRMATORY_RUN_AUTHORIZED": True,
        "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
        "SYNTHETIC_CONFIRMATORY_COMPLETE": False,
        "SYNTHETIC_CONFIRMATORY_PASS": "NOT_EVALUATED",
        "REAL_DATA_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
    }


def _bind_artificial_reports(primary, independent):
    raw_sha = "a" * 64
    primary = copy.deepcopy(primary)
    independent = copy.deepcopy(independent)
    for report in (primary, independent):
        report["run_id"] = "artificial-fixture"
        report["raw_result_manifest_sha256"] = raw_sha
    return primary, independent, {
        "run_id": "artificial-fixture",
        "raw_result_manifest_sha256": raw_sha,
    }


def _prerun_evidence(decision: dict[str, object]) -> dict[str, dict[str, object]]:
    digest = "a" * 64
    commit = "b" * 40
    model_rows = [
        {
            "feature_names": ["artificial_feature"],
            "feature_order_pass": True,
            "model_id": f"artificial-model-{index}",
            "parameters_complete": True,
            "ridge_schema_pass": True,
            "training_data_sha256": digest,
        }
        for index in range(4)
    ]
    prediction_rows = [
        {
            "absolute_difference": 0.0,
            "feature_fixture": {"artificial_feature": 1.0},
            "model_id": f"artificial-model-{index}",
            "prediction_pass": True,
        }
        for index in range(4)
    ]
    suite = {
        "errors": 0,
        "expected_tests": 4,
        "failures": 0,
        "passed": 4,
        "sha256": digest,
        "skipped": 0,
        "test_pass": True,
        "tests": 4,
    }
    pcl = {
        "failed": 0,
        "expected_tests": 3,
        "passed": 3,
        "sha256": digest,
        "skipped": 0,
        "test_names": ["identity", "transform", "degeneracy"],
        "test_pass": True,
        "tests": 3,
    }
    implementation = {
        "bound_files": {"artificial": {"path": "frozen/artificial", "sha256": digest}},
        "formal_execution_authorized": True,
        "formal_manifest_file_sha256": digest,
        "formal_manifest_path": "frozen_assets/synthetic_confirmatory_formal_manifest_v1.json",
        "formal_manifest_payload_sha256": digest,
        "formal_output_dir": "results/synthetic_confirmatory_v1",
        "formal_run_id": "synthetic-confirmatory-v1",
        "formal_workers": 2,
        "forbidden_runner_arguments": [
            "--override", "--replace-seed", "--change-gate", "--change-model",
            "--fit-model", "--backend-subset", "--exclude-scene", "--native",
            "--ignore-manifest",
        ],
        "runner_arguments": [
            "--manifest", "--run-id", "--output-dir", "--workers", "--resume",
            "--dry-run",
        ],
        "schema_version": "synthetic_confirmatory_prerun_implementation_manifest_v1",
        "scientific_survival_commit": commit,
        "scientific_survival_tag": "archive/artificial-survival",
    }
    return {
        "protocol_binding.json": {
            "CONFIRMATORY_PROTOCOL_BINDING_PASS": True,
            "SCIENTIFIC_SURVIVAL_BINDING_PASS": True,
            "SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS": True,
            "expected_file_sha256": {"artificial": digest},
            "file_sha256": {"artificial": digest},
            "file_sha256_pass": True,
            "protocol_binding_checks": {"artificial_binding": True},
            "protocol_payload_sha256_computed": digest,
            "protocol_payload_sha256_recorded": digest,
            "scientific_survival_commit": commit,
            "scientific_survival_file_binding_pass": True,
            "scientific_survival_tag": "archive/artificial-survival",
        },
        "gate_contract_audit.json": {
            "CONFIRMATORY_GATE_CONTRACT_PASS": True,
            "computed_gate_contract_payload_sha256": digest,
            "exact_hypothesis_set": True,
            "hypothesis_count": 6,
            "hypothesis_results": {f"H{index}": True for index in range(1, 7)},
            "recorded_gate_contract_payload_sha256": digest,
        },
        "seed_provenance_audit.json": {
            "CONFIRMATORY_RNG_INSTANTIATION_COUNT": 0,
            "CONFIRMATORY_SEED_INSTANTIATION_COUNT": 0,
            "CONFIRMATORY_SEED_PARSE_ERROR_COUNT": 0,
            "CONFIRMATORY_SEED_PROVENANCE_PASS": True,
            "CONFIRMATORY_SEED_USAGE_HIT_COUNT": 0,
            "stored_provenance_pass": True,
            "structured_declaration_mention_count": 1,
            "structured_file_count": 1,
            "structured_usage_hits": [],
        },
        "plan_audit.json": {
            "CONFIRMATORY_INDEPENDENT_NO_PSEUDOREPLICATION_PASS": True,
            "CONFIRMATORY_PLAN_COUNT_PASS": True,
            "CONFIRMATORY_PLAN_PAIRING_PASS": True,
            "CONFIRMATORY_PLAN_UNIQUENESS_PASS": True,
            "backend_trial_counts": {
                "open3d_point_to_plane": 595,
                "pcl_point_to_plane": 595,
            },
            "condition_snapshot_counts": {
                "FULL_NOISE": 525,
                "IDEAL_MATCHED": 35,
                "INDEPENDENT_NOISE_FREE": 35,
            },
            "duplicate_snapshot_count": 0,
            "duplicate_trial_count": 0,
            "full_noise_replicates_per_scene_geometry": 15,
            "independent_pseudoreplication_plan_count": 0,
            "native_trial_count": 0,
            "pairing_violation_count": 0,
            "planned_snapshot_count": 595,
            "planned_snapshot_identity_sha256": digest,
            "planned_snapshot_unique_count": 595,
            "planned_trial_count": 1190,
            "planned_trial_identity_sha256": digest,
            "planned_trial_unique_count": 1190,
            "semantic_violation_count": 0,
            "snapshot_id_formula_mismatch_count": 0,
            "trial_id_formula_mismatch_count": 0,
            "trial_metadata_mismatch_count": 0,
        },
        "frozen_model_audit.json": {
            "FROZEN_MODEL_FEATURE_ORDER_PASS": True,
            "FROZEN_MODEL_FILE_SHA_PASS": True,
            "FROZEN_MODEL_NO_REFIT_CONTRACT_PASS": True,
            "FROZEN_MODEL_SCHEMA_PASS": True,
            "MODEL_FIT_CALL_COUNT": 0,
            "SCALER_FIT_CALL_COUNT": 0,
            "actual_model_file_sha256": digest,
            "expected_model_file_sha256": digest,
            "forbidden_fit_calls": [],
            "model_count": 4,
            "models": model_rows,
            "payload_sha256_pass": True,
            "training_code_sha256": {
                "feature_source_sha256": digest,
                "feature_source_sha256_pass": True,
                "survival_model_code_sha256": digest,
                "survival_model_code_sha256_pass": True,
            },
        },
        "frozen_model_prediction_crosscheck.json": {
            "FROZEN_MODEL_PREDICTION_CROSSCHECK_PASS": True,
            "MODEL_FIT_CALL_COUNT": 0,
            "SCALER_FIT_CALL_COUNT": 0,
            "absolute_tolerance": 1.0e-12,
            "fixture_rng_or_seed_use_count": 0,
            "maximum_absolute_prediction_difference": 0.0,
            "models": prediction_rows,
        },
        "fixture_regression_report.json": {
            "CONFIRMATORY_EXECUTION_CHAIN_FIXTURE_PASS": True,
            "FIXTURE_QUALIFICATION_PASS": True,
            "analysis_verifier_difference_count": 0,
            "artifact_verifier_pass": True,
            "backend_execution_count": 6,
            "confirmatory_seed_or_rng_use_count": 0,
            "fixture_backend_execution_is_confirmatory_count": 0,
            "fixture_snapshot_count": 3,
            "fixture_trial_count": 6,
            "fresh_resume_scientific_equivalence": True,
            "input_pairing_violation_count": 0,
            "publisher_pass": True,
            "resume_backend_execution_count": 0,
        },
        "dry_run_report.json": {
            **{name: 0 for name in PRERUN_ZERO_EXECUTION_COUNTER_NAMES},
            "CONFIRMATORY_DRY_RUN_PASS": True,
            "SYNTHETIC_CONFIRMATORY_EXECUTED": False,
            "attempt_started_event_count": 0,
            "confirmatory_backend_execution_count": 0,
            "confirmatory_rng_instantiation_count": 0,
            "confirmatory_snapshot_generation_count": 0,
            "confirmatory_trial_result_count": 0,
            "condition_snapshot_counts": {
                "IDEAL_MATCHED": 35,
                "INDEPENDENT_NOISE_FREE": 35,
                "FULL_NOISE": 525,
            },
            "duplicate_snapshot_count": 0,
            "duplicate_trial_count": 0,
            "formal_execution_authorized_before_freeze": False,
            "independent_pseudoreplication_plan_count": 0,
            "native_trial_count": 0,
            "open3d_trial_count": 595,
            "output_dir": "results/synthetic_confirmatory_v1",
            "output_dir_created": False,
            "pairing_violation_count": 0,
            "pcl_trial_count": 595,
            "planned_snapshot_count": 595,
            "planned_trial_count": 1190,
            "run_id": "synthetic-confirmatory-v1",
            "schema_version": "synthetic_confirmatory_dry_run_v1",
            "workers": 2,
        },
        "implementation_manifest.json": implementation,
        "test_report.json": {
            "PCL_V3_FIXTURE_QUALIFICATION_PASS": True,
            "SOURCE_DEGEN_LIO_PYTEST_EXECUTED": False,
            "full_harness": dict(suite),
            "pcl_backend_v3": pcl,
            "schema_version": "synthetic_confirmatory_prerun_test_report_v1",
            "specialized_confirmatory": dict(suite),
            "test_report_pass": True,
        },
    }


def _refresh_prerun_sha(root: Path) -> None:
    listed = sorted(
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(root).as_posix()
        not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in listed
        ),
        encoding="utf-8",
    )


def _test_fixture_publication_binding(artifact: Path) -> dict[str, object]:
    source = ROOT / "artifacts/fixture_qualification"
    fixture_run = artifact.parent / "fixture-run"
    fixture_run.mkdir()
    shutil.copytree(source / "raw_results", fixture_run / "raw_results")
    raw = json.loads(
        (source / "raw_result_manifest.json").read_text(encoding="utf-8")
    )
    manifest_path = ROOT / "frozen_assets/frozen_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in raw["results"].values():
        result_path = fixture_run / "raw_results" / entry["path"]
        row = json.loads(result_path.read_text(encoding="utf-8"))
        row["implementation_sha256"] = manifest["manifest_payload_sha256"]
        result_path.write_bytes(canonical_json_bytes(row))
        entry["sha256"] = file_sha256(result_path)
    (fixture_run / "raw_result_manifest.json").write_bytes(
        canonical_json_bytes(raw)
    )
    publication = artifact / "fixture_publication"
    audit_and_publish_existing_fixture_results(
        manifest_path=manifest_path,
        fixture_run_dir=fixture_run,
        artifact_dir=publication,
    )
    live = verify_fixture_publication_artifact(publication, write_report=False)
    assert live["FIXTURE_ARTIFACT_VERIFICATION_PASS"] is True
    inventory = [
        {
            "path": candidate.relative_to(publication).as_posix(),
            "sha256": file_sha256(candidate),
        }
        for candidate in sorted(
            (path for path in publication.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(publication).as_posix(),
        )
    ]
    encoded = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "artifact_relative_path": "fixture_publication",
        "artifact_verification_file_sha256": file_sha256(
            publication / "artifact_verification.json"
        ),
        "file_count": len(inventory),
        "file_inventory": inventory,
        "file_inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "live_verification": live,
        "schema_version": "synthetic_confirmatory_fixture_publication_binding_v1",
        "sha256sums_file_sha256": file_sha256(publication / "SHA256SUMS"),
    }


def _prerun_artifact(root: Path) -> None:
    root.mkdir()
    decision = _prerun_decision()
    evidence = _prerun_evidence(decision)
    evidence["fixture_regression_report.json"][
        "fixture_publication_binding"
    ] = _test_fixture_publication_binding(root)
    values = {
        **evidence,
        "final_decision.json": decision,
        "run_manifest.json": {
            "confirmatory_formal_backend_execution_count": 0,
            "confirmatory_formal_snapshot_count": 0,
            "confirmatory_formal_trial_result_count": 0,
            "final_decision": decision,
            "fixture_backend_execution_count": 6,
            "fixture_snapshot_count": 3,
            "fixture_trial_count": 6,
            "formal_manifest_payload_sha256": "a" * 64,
            "formal_run_id": "synthetic-confirmatory-v1",
            "pcl_v3_fixture_test_count": 3,
            "schema_version": "synthetic_confirmatory_prerun_run_manifest_v1",
            "source_degen_lio_pytest_executed": False,
        },
    }
    for name, value in values.items():
        (root / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (root / "pre_run_report.md").write_text(
        "SYNTHETIC_CONFIRMATORY_PRE_RUN_QUALIFICATION_PASS\n"
        "CONFIRMATORY_RUN_AUTHORIZED\n"
        "SYNTHETIC_CONFIRMATORY_EXECUTED\n"
        "SYNTHETIC_CONFIRMATORY_PASS\n"
        "no Confirmatory snapshot\n",
        encoding="utf-8",
    )
    _refresh_prerun_sha(root)


def test_prerun_artifact_verifier_is_write_read_stable(tmp_path: Path):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    written = verify_synthetic_confirmatory_prerun_artifact(
        artifact, write_report=True
    )
    reread = verify_synthetic_confirmatory_prerun_artifact(
        artifact, write_report=False
    )
    assert written == reread
    assert written["actual_file_count"] == 33
    assert written["fixture_publication_file_count"] == 18
    assert written["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is True


def test_prerun_verifier_rejects_self_reported_fixture_pass_without_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    shutil.rmtree(artifact / "fixture_publication")
    fixture_path = artifact / "fixture_regression_report.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture.pop("fixture_publication_binding")
    fixture["publisher_pass"] = True
    fixture["artifact_verifier_pass"] = True
    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_prerun_sha(artifact)
    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["fixture_publication_binding_pass"] is False
    assert report["fixture_publication_live_verification_pass"] is False
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_prerun_verifier_rejects_gate_or_counter_tamper(tmp_path: Path):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    decision_path = artifact / "final_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision[PRERUN_REQUIRED_GATE_NAMES[0]] = False
    decision_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["required_gate_value_pass"] is False
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


@pytest.mark.parametrize(
    ("file_name", "field", "tampered", "failure_fragment"),
    [
        (
            "protocol_binding.json",
            "SYNTHETIC_CONFIRMATORY_PROTOCOL_CONTRACT_AUDIT_PASS",
            False,
            "protocol_binding.json:overall_contract",
        ),
        (
            "gate_contract_audit.json",
            "hypothesis_count",
            5,
            "gate_contract_audit.json:hypotheses",
        ),
        (
            "seed_provenance_audit.json",
            "CONFIRMATORY_SEED_USAGE_HIT_COUNT",
            1,
            "seed_provenance_audit.json:zero_seed_use",
        ),
        (
            "plan_audit.json",
            "planned_snapshot_count",
            594,
            "plan_audit.json:cardinality",
        ),
        (
            "frozen_model_audit.json",
            "MODEL_FIT_CALL_COUNT",
            1,
            "frozen_model_audit.json:no_fit",
        ),
        (
            "frozen_model_prediction_crosscheck.json",
            "maximum_absolute_prediction_difference",
            1.0e-6,
            "frozen_model_prediction_crosscheck.json:predictions",
        ),
        (
            "fixture_regression_report.json",
            "fixture_trial_count",
            5,
            "fixture_regression_report.json:qualification",
        ),
        (
            "dry_run_report.json",
            "CONFIRMATORY_BACKEND_EXECUTION_COUNT",
            1,
            "dry_run_report.json:no_execution",
        ),
        (
            "implementation_manifest.json",
            "formal_workers",
            3,
            "implementation_manifest.json:identity",
        ),
        (
            "test_report.json",
            "test_report_pass",
            False,
            "test_report.json:overall",
        ),
    ],
)
def test_prerun_verifier_semantically_rejects_each_evidence_tamper(
    tmp_path: Path,
    file_name: str,
    field: str,
    tampered: object,
    failure_fragment: str,
):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    path = artifact / file_name
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = tampered
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_prerun_sha(artifact)

    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["sha256_mismatch_files"] == []
    assert report["evidence_semantic_crosscheck_pass"] is False
    assert failure_fragment in report["evidence_semantic_failures"]
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_prerun_verifier_rejects_empty_evidence_even_with_matching_sha(
    tmp_path: Path,
):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    empty_name = "plan_audit.json"
    (artifact / empty_name).write_text("{}\n", encoding="utf-8")
    _refresh_prerun_sha(artifact)

    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["sha256_mismatch_files"] == []
    assert report["empty_evidence_files"] == [empty_name]
    assert report["evidence_semantic_crosscheck_pass"] is False
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_prerun_verifier_rejects_run_manifest_crosscheck_tamper(
    tmp_path: Path,
):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    path = artifact / "run_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["confirmatory_formal_trial_result_count"] = 1
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_prerun_sha(artifact)

    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["sha256_mismatch_files"] == []
    assert "run_manifest.json:no_formal_execution" in report[
        "evidence_semantic_failures"
    ]
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_prerun_verifier_rejects_skipped_or_incomplete_test_suite(
    tmp_path: Path,
):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    path = artifact / "test_report.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["specialized_confirmatory"]["skipped"] = 1
    value["specialized_confirmatory"]["passed"] = 3
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_prerun_sha(artifact)
    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert "test_report.json:specialized" in report["evidence_semantic_failures"]
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_prerun_verifier_rejects_extra_file(tmp_path: Path):
    artifact = tmp_path / "prerun"
    _prerun_artifact(artifact)
    (artifact / "performance.csv").write_text("forbidden\n", encoding="utf-8")
    report = verify_synthetic_confirmatory_prerun_artifact(artifact)
    assert report["extra_files"] == ["performance.csv"]
    assert report["CONFIRMATORY_ARTIFACT_VERIFICATION_PASS"] is False


def test_seed_free_internal_publisher_and_formal_verifier(
    tmp_path: Path, evidence, gate_contract
):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    primary, independent, run_manifest = _bind_artificial_reports(
        primary, independent
    )
    destination = tmp_path / "formal-artifact"
    verification = _publish_into(
        destination,
        primary=primary,
        independent=independent,
        run_manifest=run_manifest,
    )
    assert verification["ARTIFACT_VERIFICATION_PASS"] is True
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*") if path.is_file()
    } == set(FORMAL_REQUIRED_FILES)
    assert verify_synthetic_confirmatory_artifact(destination) == verification


def test_formal_artifact_verifier_rejects_tamper(
    tmp_path: Path, evidence, gate_contract
):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    primary, independent, run_manifest = _bind_artificial_reports(
        primary, independent
    )
    destination = tmp_path / "formal-artifact"
    _publish_into(
        destination,
        primary=primary,
        independent=independent,
        run_manifest=run_manifest,
    )
    with (destination / "tables/h1_ideal_control.csv").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    report = verify_synthetic_confirmatory_artifact(destination)
    assert report["ARTIFACT_VERIFICATION_PASS"] is False
    assert report["sha256_mismatch_files"] == ["tables/h1_ideal_control.csv"]


def test_publisher_rejects_analysis_not_bound_to_live_raw_identity(
    tmp_path: Path, evidence, gate_contract
):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    primary, independent, run_manifest = _bind_artificial_reports(
        primary, independent
    )
    independent["raw_result_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="live raw evidence"):
        _publish_into(
            tmp_path / "rejected-artifact",
            primary=primary,
            independent=independent,
            run_manifest=run_manifest,
        )


def test_publisher_rejects_subtolerance_numeric_difference(
    tmp_path: Path, evidence, gate_contract
):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    primary, independent, run_manifest = _bind_artificial_reports(
        primary, independent
    )
    independent["verification_projection"]["h3_cross_backend_ranking"][0][
        "spearman_rho"
    ] -= 5.0e-13
    with pytest.raises(ValueError, match="analysis differs"):
        _publish_into(
            tmp_path / "rejected-artifact",
            primary=primary,
            independent=independent,
            run_manifest=run_manifest,
        )


def test_formal_artifact_verifier_rejects_raw_binding_tamper_with_fresh_sha(
    tmp_path: Path, evidence, gate_contract
):
    trials, common = evidence
    primary = _primary(evidence, gate_contract)
    independent = independently_recompute_synthetic_confirmatory(
        trials=trials,
        common_records=common,
        model_lock=_models(),
        gate_contract=gate_contract,
        expected_geometry_seeds=GEOMETRIES,
    )
    primary, independent, run_manifest = _bind_artificial_reports(
        primary, independent
    )
    destination = tmp_path / "formal-artifact"
    _publish_into(
        destination,
        primary=primary,
        independent=independent,
        run_manifest=run_manifest,
    )
    path = destination / "independent_verification.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["raw_result_manifest_sha256"] = "b" * 64
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    listed = sorted(
        candidate for candidate in destination.rglob("*")
        if candidate.is_file()
        and candidate.name not in {"SHA256SUMS", "artifact_verification.json"}
    )
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(candidate.read_bytes()).hexdigest()}  "
            f"{candidate.relative_to(destination).as_posix()}\n"
            for candidate in listed
        ),
        encoding="utf-8",
    )
    report = verify_synthetic_confirmatory_artifact(destination)
    assert report["sha256_mismatch_files"] == []
    assert report["analysis_input_binding_pass"] is False
    assert report["ARTIFACT_VERIFICATION_PASS"] is False
