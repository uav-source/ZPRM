import math

import numpy as np
import pytest

from experiments.mid360_formal_batch1.postrun_verification.independent_postrun_verifier_v1 import (
    load_contract,
    recompute_transforms,
)
from experiments.mid360_formal_batch1.postrun_verification.independent_transform_math_v1 import (
    IndependentTransformError,
    independent_rotation_audit,
    independent_transform_audit,
)


IDENTITY = np.eye(4).tolist()


def result_for(matrix: np.ndarray) -> dict:
    audit = independent_transform_audit(IDENTITY, matrix)
    return {
        "finite_result": True,
        "T0": IDENTITY,
        "T_est": matrix.tolist(),
        "Delta_T": audit.delta.tolist(),
        "translation_x_m": float(audit.translation[0]),
        "translation_y_m": float(audit.translation[1]),
        "translation_z_m": float(audit.translation[2]),
        "translation_norm_m": audit.translation_norm_m,
        "rotation_angle_rad": audit.rotation.angle_rad,
        "rotation_angle_deg": audit.rotation.angle_deg,
    }


def run_result(row: dict) -> dict:
    record = {"result": row, "schema_pass": True, "identity_binding_pass": True}
    report = recompute_transforms([record], load_contract())
    return {"report": report, "record": record}


def test_identity_with_known_translation_fixture() -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = [0.1, -0.2, 0.3]
    audit = independent_transform_audit(IDENTITY, matrix)
    assert np.array_equal(audit.delta, matrix)
    assert np.allclose(audit.translation, [0.1, -0.2, 0.3])
    assert audit.translation_norm_m == pytest.approx(math.sqrt(0.14))


@pytest.mark.parametrize("angle", [2.0e-9, 1.0e-8, 0.2, math.pi])
def test_known_rotation_fixture(angle: float) -> None:
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle), 0.0],
         [math.sin(angle), math.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    audit = independent_rotation_audit(matrix)
    assert audit.angle_rad == pytest.approx(angle, abs=1e-12)
    assert audit.angle_deg == pytest.approx(math.degrees(angle), abs=1e-9)


def test_reflected_rotation_is_rejected() -> None:
    with pytest.raises(IndependentTransformError, match="quality gates"):
        independent_rotation_audit(np.diag([-1.0, 1.0, 1.0]))


def test_translation_norm_tamper_fails_closed() -> None:
    matrix = np.eye(4); matrix[:3, 3] = [0.1, 0.2, 0.3]
    row = result_for(matrix); row["translation_norm_m"] += 1e-5
    outcome = run_result(row)
    assert not outcome["report"]["pass"]
    assert not outcome["record"]["transform_arithmetic_pass"]


def test_delta_t_tamper_fails_closed() -> None:
    row = result_for(np.eye(4)); row["Delta_T"][0][3] = 1e-4
    outcome = run_result(row)
    assert outcome["report"]["Delta_T_mismatch_count"] == 1


def test_rotation_rad_tamper_fails_closed() -> None:
    row = result_for(np.eye(4)); row["rotation_angle_rad"] = 1e-5
    outcome = run_result(row)
    assert outcome["report"]["rotation_recomputation_mismatch_count"] == 1


def test_rotation_deg_tamper_fails_closed() -> None:
    row = result_for(np.eye(4)); row["rotation_angle_deg"] = 1e-4
    outcome = run_result(row)
    assert outcome["report"]["rotation_recomputation_mismatch_count"] == 1


def test_nonfinite_terminal_requires_null_pose_fields() -> None:
    row = {"finite_result": False, **{name: None for name in (
        "T_est", "Delta_T", "translation_x_m", "translation_y_m", "translation_z_m",
        "translation_norm_m", "rotation_angle_rad", "rotation_angle_deg")}}
    outcome = run_result(row)
    assert outcome["report"]["valid_nonfinite_terminal_count"] == 1
    row["translation_norm_m"] = 0.0
    assert not run_result(row)["record"]["transform_arithmetic_pass"]
