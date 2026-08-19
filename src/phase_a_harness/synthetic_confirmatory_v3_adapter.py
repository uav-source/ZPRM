"""Seed-free qualification adapter for Synthetic Confirmatory v3.

This module deliberately contains no formal Confirmatory namespace or seed
value.  It only delegates the fixed three-snapshot/six-trial fixture to the
already-qualified runtime lifecycle and v2 scientific publication cores.  A
successful report from this adapter is a prerequisite for declaring the v3
seed schedule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import file_sha256
from .runtime_lifecycle_io import atomic_create_canonical_json
from .runtime_path_policy import RuntimePathLayout, qualify_runtime_paths


ADAPTER_SCHEMA = "synthetic_confirmatory_v3_execution_adapter_fixture_v1"
QUALIFICATION_RUNTIME_ROOT = Path("/home/lj/ZPRM/zero_perturbation_runtime")
QUALIFICATION_RUN_KIND = "qualification"
QUALIFICATION_RUN_ID = "synthetic_confirmatory_v3_prerun"


def qualification_layout(*, repository: str | Path, resume: bool) -> RuntimePathLayout:
    """Return the one canonical v3 pre-run fixture layout without creating it."""

    qualified = qualify_runtime_paths(
        QUALIFICATION_RUN_ID,
        runtime_root=QUALIFICATION_RUNTIME_ROOT,
        repository_root=repository,
        run_kind=QUALIFICATION_RUN_KIND,
        resume=resume,
    )
    expected = (
        QUALIFICATION_RUNTIME_ROOT
        / QUALIFICATION_RUN_KIND
        / QUALIFICATION_RUN_ID
    )
    if qualified.layout.run_root != expected:
        raise ValueError("v3 adapter qualification root differs from the contract")
    return qualified.layout


def _file_inventory(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_seed_free_adapter_fixture(
    *,
    repository: str | Path,
    expected_commit: str,
    expected_branch: str,
    expected_tag: str,
    git_gate: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run fresh/resume, analysis, publication, and verification externally.

    The caller owns the Git gate implementation so every checkpoint can be
    persisted by the supervising process.  No formal v3 asset is read here.
    """

    root = Path(repository).resolve()
    fresh_layout = qualification_layout(repository=root, resume=False)

    from .runtime_lifecycle_fixture import (
        load_completed_fixture_results,
        publish_fixture_runtime_artifact,
        run_fixture_lifecycle,
    )
    from .synthetic_confirmatory_v2_analysis import analyze_v2_fixture_results
    from .synthetic_confirmatory_v2_artifact_verifier import (
        verify_synthetic_confirmatory_v2_fixture_artifact,
    )
    from .synthetic_confirmatory_v2_independent_verifier import (
        compare_v2_fixture_primary_and_independent,
        independently_analyze_v2_fixture_results,
    )

    git_gate("V3_FIXTURE_PRE_RUN_GIT_GATE")
    fresh = run_fixture_lifecycle(
        repository=root,
        layout=fresh_layout,
        run_id=QUALIFICATION_RUN_ID,
        invocation_id="fresh",
        workers=2,
        resume=False,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        runtime_path_policy_sha256=file_sha256(
            root / "src/phase_a_harness/runtime_path_policy.py"
        ),
        qualification_delay_seconds=0.0,
        git_gate=git_gate,
    )
    snapshot_before = _file_inventory(fresh_layout.snapshot_cache)
    trial_before = _file_inventory(fresh_layout.raw_results)

    resume_layout = qualification_layout(repository=root, resume=True)
    git_gate("V3_FIXTURE_RESUME_GIT_GATE")
    resumed = run_fixture_lifecycle(
        repository=root,
        layout=resume_layout,
        run_id=QUALIFICATION_RUN_ID,
        invocation_id="resume",
        workers=2,
        resume=True,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tag=expected_tag,
        runtime_path_policy_sha256=file_sha256(
            root / "src/phase_a_harness/runtime_path_policy.py"
        ),
        qualification_delay_seconds=0.0,
        git_gate=git_gate,
    )
    snapshot_after = _file_inventory(resume_layout.snapshot_cache)
    trial_after = _file_inventory(resume_layout.raw_results)

    rows = load_completed_fixture_results(
        repository=root,
        layout=resume_layout,
        run_id=QUALIFICATION_RUN_ID,
        contract_sha256=resumed["run_contract_sha256"],
        implementation_sha256=(
            json.loads(
                (root / "frozen_assets/frozen_experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["manifest_payload_sha256"]
        ),
    )
    primary = analyze_v2_fixture_results(rows)
    independent = independently_analyze_v2_fixture_results(rows)
    difference = compare_v2_fixture_primary_and_independent(primary, independent)
    resume_layout.primary_analysis.mkdir(parents=True, exist_ok=False)
    resume_layout.independent_verification.mkdir(parents=True, exist_ok=False)
    atomic_create_canonical_json(
        resume_layout.primary_analysis / "primary_analysis.json", primary
    )
    atomic_create_canonical_json(
        resume_layout.independent_verification / "independent_verification.json",
        independent,
    )
    atomic_create_canonical_json(
        resume_layout.independent_verification
        / "primary_independent_difference.json",
        difference,
    )
    git_gate("V3_FIXTURE_POST_ANALYSIS_GIT_GATE")

    publication = publish_fixture_runtime_artifact(
        layout=resume_layout,
        primary=primary,
        independent=independent,
        run_manifest={
            "schema_version": "synthetic_confirmatory_v2_fixture_run_v1",
            "backend_execution_count": 6,
            "fixture_snapshot_count": 3,
            "fixture_trial_count": 6,
            "formal_confirmatory_science_evaluated": False,
            "formal_v2_seed_reference_count": 0,
            "fresh_resume_scientific_equivalence": (
                snapshot_before == snapshot_after and trial_before == trial_after
            ),
            "resume_backend_execution_count": 0,
        },
    )
    artifact_path = Path(publication["artifact_staging_path"])
    artifact_verification = verify_synthetic_confirmatory_v2_fixture_artifact(
        artifact_path, write_report=False
    )
    git_gate("V3_FIXTURE_POST_PUBLISHER_GIT_GATE")

    report = {
        "schema_version": ADAPTER_SCHEMA,
        "V3_EXECUTION_ADAPTER_FIXTURE_PASS": bool(
            fresh.get("FIXTURE_EXECUTION_CHAIN_PASS") is True
            and fresh.get("fixture_snapshot_count") == 3
            and fresh.get("fixture_trial_count") == 6
            and fresh.get("backend_trial_counts")
            == {"open3d_point_to_plane": 3, "pcl_point_to_plane": 3}
            and fresh.get("native_execution_count") == 0
            and resumed.get("generated_snapshot_count") == 0
            and resumed.get("backend_execution_count_this_invocation") == 0
            and resumed.get("resume_skipped_valid_snapshot_count") == 3
            and resumed.get("resume_skipped_valid_result_count") == 6
            and snapshot_before == snapshot_after
            and trial_before == trial_after
            and difference.get("exact_match_pass") is True
            and artifact_verification.get(
                "FIXTURE_ARTIFACT_VERIFICATION_PASS"
            )
            is True
        ),
        "V3_RUNTIME_PATH_POLICY_PASS": True,
        "V3_PRIMARY_VERIFIER_FIXTURE_PASS": difference.get("exact_match_pass")
        is True,
        "V3_PUBLISHER_FIXTURE_PASS": publication.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        ) is True,
        "V3_ARTIFACT_VERIFIER_FIXTURE_PASS": artifact_verification.get(
            "FIXTURE_ARTIFACT_VERIFICATION_PASS"
        )
        is True,
        "formal_confirmatory_science_evaluated": False,
        "formal_seed_reference_count": 0,
        "formal_seed_rng_instantiation_count": 0,
        "fresh": fresh,
        "resume": resumed,
        "valid_snapshot_reexecution_count": 0,
        "valid_trial_reexecution_count": 0,
        "snapshot_checksum_change_after_resume": int(
            snapshot_before != snapshot_after
        ),
        "trial_checksum_change_after_resume": int(trial_before != trial_after),
        "primary": primary,
        "independent": independent,
        "primary_independent_difference": difference,
        "publication": publication,
        "artifact_verification": artifact_verification,
        "runtime_root": str(resume_layout.run_root),
    }
    if report["V3_EXECUTION_ADAPTER_FIXTURE_PASS"] is not True:
        raise RuntimeError("v3 seed-free adapter fixture qualification failed")
    return report


__all__ = [
    "ADAPTER_SCHEMA",
    "QUALIFICATION_RUN_ID",
    "QUALIFICATION_RUNTIME_ROOT",
    "qualification_layout",
    "run_seed_free_adapter_fixture",
]
