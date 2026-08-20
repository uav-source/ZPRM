"""Deterministic fixture-only qualification of the future FMB1 run lifecycle.

The 360 cells produced here are schema-like lifecycle fixtures.  They contain
no point clouds, no real scene identifiers, no backend outputs, and no
registration call.  Nothing from this module is measurement evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FIXTURE_CLASSIFICATION = "FIXTURE_ONLY_DO_NOT_CITE"
FIXTURE_SCENES = tuple(f"FIXTURE_SCENE_{index:02d}" for index in range(1, 7))
FIXTURE_STATIONS = tuple(f"FIXTURE_STATION_{index:02d}" for index in range(1, 4))
FIXTURE_SNAPSHOTS = tuple(f"FIXTURE_Q{index:02d}" for index in range(1, 11))
FIXTURE_BACKENDS = (
    "OPEN3D_SCHEMA_ADAPTER_FIXTURE",
    "PCL_SCHEMA_ADAPTER_FIXTURE",
)
PLANNED_SNAPSHOT_COUNT = 180
PLANNED_OPEN3D_TRIAL_COUNT = 180
PLANNED_PCL_TRIAL_COUNT = 180
PLANNED_TRIAL_COUNT = 360
ACTUAL_REGISTRATION_EXECUTION = False

ROOT_MARKER_NAME = "FIXTURE_ONLY_ROOT.json"
MANIFEST_NAME = "fixture_trial_manifest.json"
INDEX_NAME = "fixture_result_index.json"
COMPLETION_NAME = "fixture_completion.json"
RESULT_DIRECTORY = "fixture_results"

COMMON_FLAGS = {
    "classification": FIXTURE_CLASSIFICATION,
    "FIXTURE_ONLY": True,
    "FIXTURE_ONLY_DO_NOT_CITE": True,
    "NOT_REAL_FMB1": True,
    "NOT_FORMAL_MEASUREMENT": True,
    "ACTUAL_REGISTRATION_EXECUTION": ACTUAL_REGISTRATION_EXECUTION,
    "NO_FORMAL_REGISTRATION": True,
    "FORMAL_REGISTRATION_AUTHORIZED": False,
    "FORMAL_ICP_UNLOCKED": False,
    "FORMAL_EXECUTION_UNLOCKED": False,
    "READY_FOR_SEPARATE_FORMAL_REGISTRATION_AUTHORIZATION": False,
    "MEASUREMENT_FINAL_RESULT": False,
}


class FixtureExecutionError(RuntimeError):
    """Raised when fixture lifecycle state is missing, partial, or tampered."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_digest(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[digest_field] = _canonical_sha256(output)
    return output


def _validate_digest(
    payload: Mapping[str, Any], digest_field: str, label: str
) -> None:
    declared = payload.get(digest_field)
    material = dict(payload)
    material.pop(digest_field, None)
    if not isinstance(declared, str) or declared != _canonical_sha256(material):
        raise FixtureExecutionError(f"{label} checksum mismatch")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FixtureExecutionError(f"stale partial file exists: {partial}")
    partial.write_bytes(_canonical_bytes(payload) + b"\n")
    partial.replace(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureExecutionError(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FixtureExecutionError(f"{label} must be a JSON object")
    return payload


def _synthetic_sha(label: str) -> str:
    return hashlib.sha256(("FMB1_FIXTURE_ONLY|" + label).encode("utf-8")).hexdigest()


def build_fixture_plan() -> dict[str, Any]:
    """Return the immutable, backend-free 6x3x10x2 fixture plan."""

    trials: list[dict[str, Any]] = []
    trial_index = 0
    for scene_id in FIXTURE_SCENES:
        for station_id in FIXTURE_STATIONS:
            target_sha = _synthetic_sha(f"target|{scene_id}|{station_id}")
            for snapshot_id in FIXTURE_SNAPSHOTS:
                source_sha = _synthetic_sha(
                    f"source|{scene_id}|{station_id}|{snapshot_id}"
                )
                for backend_label in FIXTURE_BACKENDS:
                    trial_index += 1
                    trial_id = (
                        f"{scene_id}/{station_id}/{snapshot_id}::{backend_label}"
                    )
                    trials.append(
                        {
                            "trial_index": trial_index,
                            "trial_id": trial_id,
                            "scene_id": scene_id,
                            "station_id": station_id,
                            "snapshot_id": snapshot_id,
                            "backend_schema_path": backend_label,
                            "synthetic_source_sha256": source_sha,
                            "synthetic_target_sha256": target_sha,
                            "T0": "IDENTITY_4X4_FIXTURE_TOKEN",
                            "planned_only": True,
                            "actual_execution": False,
                        }
                    )
    if trial_index != PLANNED_TRIAL_COUNT:
        raise AssertionError("fixture plan cardinality drift")
    body = {
        "schema": "mid360_fmb1_prebackend_fixture_plan_v1",
        **COMMON_FLAGS,
        "scene_count": len(FIXTURE_SCENES),
        "station_count": len(FIXTURE_SCENES) * len(FIXTURE_STATIONS),
        "snapshot_count": (
            len(FIXTURE_SCENES) * len(FIXTURE_STATIONS) * len(FIXTURE_SNAPSHOTS)
        ),
        "planned_snapshot_count": PLANNED_SNAPSHOT_COUNT,
        "planned_open3d_trials": PLANNED_OPEN3D_TRIAL_COUNT,
        "planned_pcl_trials": PLANNED_PCL_TRIAL_COUNT,
        "planned_total_trials": PLANNED_TRIAL_COUNT,
        "backend_schema_path_count": len(FIXTURE_BACKENDS),
        "planned_trial_count": len(trials),
        "real_scene_access_count": 0,
        "real_backend_output_read_count": 0,
        "trials": trials,
    }
    return _with_digest(body, "manifest_sha256")


def _trial_filename(trial: Mapping[str, Any]) -> str:
    return f"T{int(trial['trial_index']):04d}.json"


def _result_payload(trial: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    body = {
        "schema": "mid360_fmb1_prebackend_fixture_trial_result_v1",
        **COMMON_FLAGS,
        "manifest_sha256": manifest_sha256,
        "trial_index": int(trial["trial_index"]),
        "trial_id": str(trial["trial_id"]),
        "scene_id": str(trial["scene_id"]),
        "station_id": str(trial["station_id"]),
        "snapshot_id": str(trial["snapshot_id"]),
        "backend_schema_path": str(trial["backend_schema_path"]),
        "synthetic_source_sha256": str(trial["synthetic_source_sha256"]),
        "synthetic_target_sha256": str(trial["synthetic_target_sha256"]),
        "T0": str(trial["T0"]),
        "lifecycle_status": "FIXTURE_TRIAL_COMPLETED_WITHOUT_BACKEND",
        "backend_invocation_count": 0,
        "scientific_result_field_count": 0,
    }
    return _with_digest(body, "payload_sha256")


def expected_fixture_state_sha256() -> str:
    """Return the complete content-state SHA without touching the filesystem."""

    plan = build_fixture_plan()
    result_rows = []
    for trial in plan["trials"]:
        payload = _result_payload(trial, str(plan["manifest_sha256"]))
        file_bytes = _canonical_bytes(payload) + b"\n"
        result_rows.append(
            {
                "trial_id": trial["trial_id"],
                "file_sha256": hashlib.sha256(file_bytes).hexdigest(),
            }
        )
    return _canonical_sha256(
        {
            "manifest_sha256": plan["manifest_sha256"],
            "results": result_rows,
        }
    )


def _empty_index(manifest_sha256: str) -> dict[str, Any]:
    return _with_digest(
        {
            "schema": "mid360_fmb1_prebackend_fixture_result_index_v1",
            **COMMON_FLAGS,
            "manifest_sha256": manifest_sha256,
            "rows": [],
        },
        "index_sha256",
    )


def _write_index(root: Path, rows: Sequence[Mapping[str, Any]], manifest_sha: str) -> None:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["trial_index"]))
    payload = _with_digest(
        {
            "schema": "mid360_fmb1_prebackend_fixture_result_index_v1",
            **COMMON_FLAGS,
            "manifest_sha256": manifest_sha,
            "rows": ordered,
        },
        "index_sha256",
    )
    _atomic_write_json(root / INDEX_NAME, payload)


def _initialize_root(root: Path) -> dict[str, Any]:
    root = root.expanduser()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FixtureExecutionError("fresh fixture root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema": "mid360_fmb1_prebackend_fixture_root_v1",
        **COMMON_FLAGS,
    }
    plan = build_fixture_plan()
    _atomic_write_json(root / ROOT_MARKER_NAME, marker)
    _atomic_write_json(root / MANIFEST_NAME, plan)
    (root / RESULT_DIRECTORY).mkdir()
    _write_index(root, [], str(plan["manifest_sha256"]))
    return plan


def _validate_root_marker(root: Path) -> None:
    marker = _read_json(root / ROOT_MARKER_NAME, "fixture root marker")
    expected = {
        "schema": "mid360_fmb1_prebackend_fixture_root_v1",
        **COMMON_FLAGS,
    }
    if marker != expected:
        raise FixtureExecutionError("fixture root marker changed")


def _load_manifest(root: Path) -> dict[str, Any]:
    payload = _read_json(root / MANIFEST_NAME, "fixture manifest")
    _validate_digest(payload, "manifest_sha256", "fixture manifest")
    expected = build_fixture_plan()
    if payload != expected:
        raise FixtureExecutionError("fixture manifest differs from immutable plan")
    return payload


def _load_index(root: Path, manifest_sha: str) -> dict[str, Any]:
    payload = _read_json(root / INDEX_NAME, "fixture result index")
    _validate_digest(payload, "index_sha256", "fixture result index")
    if payload.get("manifest_sha256") != manifest_sha:
        raise FixtureExecutionError("fixture result index manifest binding changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise FixtureExecutionError("fixture result index rows are invalid")
    ids = [str(row.get("trial_id")) for row in rows]
    paths = [str(row.get("relative_path")) for row in rows]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise FixtureExecutionError("duplicate fixture result index row")
    return payload


def _validate_result_payload(
    payload: Mapping[str, Any], trial: Mapping[str, Any], manifest_sha: str
) -> None:
    _validate_digest(payload, "payload_sha256", "fixture result payload")
    if dict(payload) != _result_payload(trial, manifest_sha):
        raise FixtureExecutionError(
            f"fixture result payload differs for {trial['trial_id']}"
        )


def _state_sha(manifest_sha: str, results: Mapping[str, Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        {
            "manifest_sha256": manifest_sha,
            "results": [
                {
                    "trial_id": trial_id,
                    "file_sha256": results[trial_id]["file_sha256"],
                }
                for trial_id in sorted(results)
            ],
        }
    )


def _completion_payload(manifest_sha: str, state_sha: str) -> dict[str, Any]:
    return _with_digest(
        {
            "schema": "mid360_fmb1_prebackend_fixture_completion_v1",
            **COMMON_FLAGS,
            "manifest_sha256": manifest_sha,
            "fixture_state_sha256": state_sha,
            "completed_trial_count": PLANNED_TRIAL_COUNT,
            "actual_registration_execution_count": 0,
        },
        "completion_sha256",
    )


def _authenticate_state(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    _validate_root_marker(root)
    partials = sorted(str(path) for path in root.rglob("*.partial"))
    if partials:
        raise FixtureExecutionError(f"partial fixture files are forbidden: {partials}")
    plan = _load_manifest(root)
    manifest_sha = str(plan["manifest_sha256"])
    trials = {str(row["trial_id"]): row for row in plan["trials"]}
    index = _load_index(root, manifest_sha)
    indexed = {str(row["trial_id"]): dict(row) for row in index["rows"]}
    if set(indexed) - set(trials):
        raise FixtureExecutionError("fixture index contains an unknown trial")

    result_dir = root / RESULT_DIRECTORY
    if not result_dir.is_dir():
        raise FixtureExecutionError("fixture result directory is missing")
    files_by_trial: dict[str, dict[str, Any]] = {}
    result_entries = sorted(result_dir.rglob("*"))
    unexpected_entries = [
        str(path)
        for path in result_entries
        if not path.is_file() or path.parent != result_dir or path.suffix != ".json"
    ]
    if unexpected_entries:
        raise FixtureExecutionError(
            f"unexpected fixture result entries: {unexpected_entries}"
        )
    for path in result_entries:
        payload = _read_json(path, "fixture result")
        trial_id = str(payload.get("trial_id"))
        trial = trials.get(trial_id)
        if trial is None:
            raise FixtureExecutionError(f"unrecognized fixture result orphan: {path.name}")
        expected_name = _trial_filename(trial)
        if path.name != expected_name:
            raise FixtureExecutionError(
                f"duplicate or noncanonical fixture result path: {path.name}"
            )
        if trial_id in files_by_trial:
            raise FixtureExecutionError(f"duplicate fixture result: {trial_id}")
        _validate_result_payload(payload, trial, manifest_sha)
        files_by_trial[trial_id] = {
            "trial_index": int(trial["trial_index"]),
            "trial_id": trial_id,
            "relative_path": f"{RESULT_DIRECTORY}/{path.name}",
            "file_sha256": _sha256_file(path),
            "payload_sha256": payload["payload_sha256"],
        }

    for trial_id, row in indexed.items():
        actual = files_by_trial.get(trial_id)
        if actual is None:
            raise FixtureExecutionError(f"indexed fixture result is missing: {trial_id}")
        if row != actual:
            raise FixtureExecutionError(f"fixture result checksum/index mismatch: {trial_id}")

    orphan_ids = sorted(set(files_by_trial) - set(indexed))
    if orphan_ids:
        raise FixtureExecutionError(
            f"fixture result orphan is not authenticated by index: {orphan_ids}"
        )

    completion_path = root / COMPLETION_NAME
    completed = completion_path.exists()
    state_sha = _state_sha(manifest_sha, files_by_trial)
    if completed:
        completion = _read_json(completion_path, "fixture completion")
        _validate_digest(completion, "completion_sha256", "fixture completion")
        expected_completion = _completion_payload(manifest_sha, state_sha)
        if completion != expected_completion:
            raise FixtureExecutionError("fixture completion state changed")
        if len(files_by_trial) != PLANNED_TRIAL_COUNT:
            raise FixtureExecutionError("completed fixture state has missing trials")

    return {
        "root": root,
        "plan": plan,
        "trials": trials,
        "index_rows": list(indexed.values()),
        "results": files_by_trial,
        "orphan_ids": [],
        "completed": completed,
        "fixture_state_sha256": state_sha,
    }


def _validate_execution_options(
    workers: int,
    interrupt_after: int | None,
    interrupt_window: str,
) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise FixtureExecutionError("workers must be a positive integer")
    if interrupt_after is not None and (
        isinstance(interrupt_after, bool)
        or not isinstance(interrupt_after, int)
        or interrupt_after < 0
    ):
        raise FixtureExecutionError("interrupt_after must be a nonnegative integer")
    if interrupt_window not in {"after_index_commit", "after_result_before_index"}:
        raise FixtureExecutionError("unsupported interruption window")


def _invocation_report(
    *,
    mode: str,
    workers: int,
    status: str,
    initial_completed: int,
    executed: int,
    skipped: int,
    recovered_orphans: Sequence[str],
    final_completed: int,
    fixture_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "mid360_fmb1_prebackend_fixture_invocation_v1",
        **COMMON_FLAGS,
        "mode": mode,
        "status": status,
        "requested_workers": workers,
        "worker_count_affects_fixture_content": False,
        "planned_trial_count": PLANNED_TRIAL_COUNT,
        "initial_completed_trial_count": initial_completed,
        "fixture_trial_materialization_count_this_invocation": executed,
        "actual_registration_execution_count_this_invocation": 0,
        "resume_skipped_authenticated_trial_count": skipped,
        "recovered_orphan_result_count": len(recovered_orphans),
        "recovered_orphan_trial_ids": list(recovered_orphans),
        "final_completed_trial_count": final_completed,
        "fixture_state_sha256": fixture_state_sha256,
        "FORMAL_EXECUTION_UNLOCKED": False,
    }


def _execute_pending(
    root: Path,
    state: Mapping[str, Any],
    *,
    mode: str,
    workers: int,
    interrupt_after: int | None,
    interrupt_window: str,
) -> dict[str, Any]:
    plan = state["plan"]
    manifest_sha = str(plan["manifest_sha256"])
    existing = dict(state["results"])
    index_rows = [dict(row) for row in state["index_rows"]]
    initial_completed = len(existing)
    skipped = initial_completed
    executed = 0
    if interrupt_after == 0:
        return _invocation_report(
            mode=mode,
            workers=workers,
            status="INTERRUPTED_FIXTURE_ONLY",
            initial_completed=initial_completed,
            executed=0,
            skipped=skipped,
            recovered_orphans=state["orphan_ids"],
            final_completed=initial_completed,
            fixture_state_sha256=_state_sha(manifest_sha, existing),
        )

    for trial in plan["trials"]:
        trial_id = str(trial["trial_id"])
        if trial_id in existing:
            continue
        payload = _result_payload(trial, manifest_sha)
        path = root / RESULT_DIRECTORY / _trial_filename(trial)
        if path.exists():
            raise FixtureExecutionError(f"refusing to overwrite fixture result: {path}")
        _atomic_write_json(path, payload)
        executed += 1
        row = {
            "trial_index": int(trial["trial_index"]),
            "trial_id": trial_id,
            "relative_path": f"{RESULT_DIRECTORY}/{path.name}",
            "file_sha256": _sha256_file(path),
            "payload_sha256": payload["payload_sha256"],
        }
        existing[trial_id] = row
        if (
            interrupt_after is not None
            and executed == interrupt_after
            and interrupt_window == "after_result_before_index"
        ):
            return _invocation_report(
                mode=mode,
                workers=workers,
                status="INTERRUPTED_FIXTURE_ONLY",
                initial_completed=initial_completed,
                executed=executed,
                skipped=skipped,
                recovered_orphans=state["orphan_ids"],
                final_completed=len(existing),
                fixture_state_sha256=_state_sha(manifest_sha, existing),
            )
        index_rows.append(row)
        _write_index(root, index_rows, manifest_sha)
        if (
            interrupt_after is not None
            and executed == interrupt_after
            and interrupt_window == "after_index_commit"
        ):
            return _invocation_report(
                mode=mode,
                workers=workers,
                status="INTERRUPTED_FIXTURE_ONLY",
                initial_completed=initial_completed,
                executed=executed,
                skipped=skipped,
                recovered_orphans=state["orphan_ids"],
                final_completed=len(existing),
                fixture_state_sha256=_state_sha(manifest_sha, existing),
            )

    state_sha = _state_sha(manifest_sha, existing)
    if len(existing) != PLANNED_TRIAL_COUNT:
        raise FixtureExecutionError("fixture execution ended with missing trials")
    completion = _completion_payload(manifest_sha, state_sha)
    completion_path = root / COMPLETION_NAME
    if completion_path.exists():
        if _read_json(completion_path, "fixture completion") != completion:
            raise FixtureExecutionError("existing fixture completion differs")
    else:
        _atomic_write_json(completion_path, completion)
    return _invocation_report(
        mode=mode,
        workers=workers,
        status="COMPLETE_FIXTURE_ONLY",
        initial_completed=initial_completed,
        executed=executed,
        skipped=skipped,
        recovered_orphans=state["orphan_ids"],
        final_completed=len(existing),
        fixture_state_sha256=state_sha,
    )


def fresh_fixture_run(
    root: Path,
    *,
    workers: int = 1,
    interrupt_after: int | None = None,
    interrupt_window: str = "after_index_commit",
) -> dict[str, Any]:
    """Create and materialize a fresh fixture-only lifecycle state."""

    _validate_execution_options(workers, interrupt_after, interrupt_window)
    _initialize_root(root)
    state = _authenticate_state(root)
    return _execute_pending(
        state["root"],
        state,
        mode="fresh",
        workers=workers,
        interrupt_after=interrupt_after,
        interrupt_window=interrupt_window,
    )


def resume_fixture_run(
    root: Path,
    *,
    workers: int = 1,
    interrupt_after: int | None = None,
    interrupt_window: str = "after_index_commit",
) -> dict[str, Any]:
    """Authenticate indexed results and resume; every orphan fails closed."""

    _validate_execution_options(workers, interrupt_after, interrupt_window)
    state = _authenticate_state(root)
    if state["completed"]:
        return _invocation_report(
            mode="resume",
            workers=workers,
            status="COMPLETE_FIXTURE_ONLY",
            initial_completed=PLANNED_TRIAL_COUNT,
            executed=0,
            skipped=PLANNED_TRIAL_COUNT,
            recovered_orphans=(),
            final_completed=PLANNED_TRIAL_COUNT,
            fixture_state_sha256=state["fixture_state_sha256"],
        )
    return _execute_pending(
        state["root"],
        state,
        mode="resume",
        workers=workers,
        interrupt_after=interrupt_after,
        interrupt_window=interrupt_window,
    )


def analyze_fixture_run(root: Path) -> dict[str, Any]:
    """Produce lifecycle counts only; no scientific or registration analysis."""

    state = _authenticate_state(root)
    if not state["completed"]:
        raise FixtureExecutionError("fixture analysis requires a complete state")
    payloads = []
    for trial_id, row in sorted(state["results"].items()):
        path = state["root"] / str(row["relative_path"])
        payloads.append(_read_json(path, f"fixture result {trial_id}"))
    return {
        "schema": "mid360_fmb1_prebackend_fixture_analysis_v1",
        **COMMON_FLAGS,
        "analysis_scope": "LIFECYCLE_STRUCTURE_ONLY_NO_SCIENTIFIC_METRICS",
        "scene_is_highest_independent_unit": True,
        "stations_are_nested_repeats": True,
        "snapshots_are_nested_repeats": True,
        "snapshots_are_independent_scenes": False,
        "dry_run_structure_endpoint_names": [
            "scene_summaries",
            "station_variance",
            "snapshot_repeated_observations",
            "cross_backend",
            "weak_vs_rich",
            "reassociation",
        ],
        "dry_run_structure_endpoints": {
            "scene_summaries": True,
            "station_variance": True,
            "snapshot_repeated_observations": True,
            "cross_backend": True,
            "weak_vs_rich": True,
            "reassociation": True,
        },
        "dry_run_structure_endpoint_count": 6,
        "scientific_values_produced": False,
        "planned_trial_count": PLANNED_TRIAL_COUNT,
        "authenticated_fixture_result_count": len(payloads),
        "backend_invocation_count": sum(
            int(row["backend_invocation_count"]) for row in payloads
        ),
        "scientific_result_field_count": sum(
            int(row["scientific_result_field_count"]) for row in payloads
        ),
        "fixture_state_sha256": state["fixture_state_sha256"],
        "status": "PASS_FIXTURE_ONLY",
    }


def build_fixture_publication_report(
    analysis: Mapping[str, Any], resume_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a non-citable publication-shape report without publishing it."""

    if analysis.get("classification") != FIXTURE_CLASSIFICATION:
        raise FixtureExecutionError("analysis is not fixture-only")
    if resume_report.get("classification") != FIXTURE_CLASSIFICATION:
        raise FixtureExecutionError("resume report is not fixture-only")
    body = {
        "schema": "mid360_fmb1_prebackend_fixture_publication_v1",
        **COMMON_FLAGS,
        "publication_performed": False,
        "citation_allowed": False,
        "measurement_claim_allowed": False,
        "analysis_sha256": _canonical_sha256(dict(analysis)),
        "resume_report_sha256": _canonical_sha256(dict(resume_report)),
        "fixture_state_sha256": analysis.get("fixture_state_sha256"),
        "status": "QUALIFIED_STRUCTURE_FIXTURE_ONLY",
    }
    return _with_digest(body, "publication_report_sha256")


def write_fixture_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one explicitly fixture-only report, refusing ambiguous payloads."""

    if payload.get("classification") != FIXTURE_CLASSIFICATION:
        raise FixtureExecutionError("refusing to write a non-fixture report")
    if payload.get("FIXTURE_ONLY_DO_NOT_CITE") is not True:
        raise FixtureExecutionError("fixture report lacks do-not-cite flag")
    _atomic_write_json(path, payload)


__all__ = [
    "ACTUAL_REGISTRATION_EXECUTION",
    "COMMON_FLAGS",
    "FIXTURE_BACKENDS",
    "FIXTURE_CLASSIFICATION",
    "FIXTURE_SCENES",
    "FIXTURE_SNAPSHOTS",
    "FIXTURE_STATIONS",
    "FixtureExecutionError",
    "PLANNED_OPEN3D_TRIAL_COUNT",
    "PLANNED_PCL_TRIAL_COUNT",
    "PLANNED_SNAPSHOT_COUNT",
    "PLANNED_TRIAL_COUNT",
    "analyze_fixture_run",
    "build_fixture_plan",
    "build_fixture_publication_report",
    "expected_fixture_state_sha256",
    "fresh_fixture_run",
    "resume_fixture_run",
    "write_fixture_report",
]
