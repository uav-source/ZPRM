#!/usr/bin/env python3
"""Audit the Exec-R2 name heuristic and emit the Exec-R3 classification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
for entry in (str(REPOSITORY), str(SOURCE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.mid360_formal_batch1.zero_perturbation_v1_1_exec_r2_lock import (  # noqa: E402
    EXECUTION_PATHS as R2_EXECUTION_PATHS,
)
from experiments.mid360_formal_batch1.zero_perturbation_v1_1_exec_r3_lock import (  # noqa: E402
    ENVIRONMENT_PATHS,
    EXECUTION_CODE_PATHS,
    RELEASE_EVIDENCE_PATHS,
    REPLACED_R2_BINDING_IDS,
    R2_LOCK_FILENAME,
    R2_RESULTS_DIR,
)


BUG_COMMIT = "f8084795b5b63d149e10282226798c070f808b81"
ATTEMPT_003_AUDIT_COMMIT = "7274b54aed145ac0884da89d597541eb47461b99"
ATTEMPT_003_AUTHORIZATION_ID = "FMB1-AUTH-9794108265e14fed8887b202eda4ed50"
ATTEMPT_003_AUTHORIZATION_SHA256 = (
    "d3db750f782fba4a2449f923ea3ab480ded43a9f21b8b8d35633417ed5354f8e"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_show(root: Path, commit: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite different audit: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def build_audit(root: Path) -> dict[str, Any]:
    lock_path = root / R2_RESULTS_DIR / R2_LOCK_FILENAME
    r2 = json.loads(lock_path.read_text(encoding="utf-8"))
    r2_bindings = r2["bindings"]
    execution_ids = set(R2_EXECUTION_PATHS) | {
        "authorization_schema", "authorization_contract"
    }
    release_ids = {
        "execution_control_patch_report", "authorization_lifecycle_test_report"
    }
    environment_ids = {
        "environment_manifest", "pcl_executable", "backend_parameter_contract"
    }
    records = []
    for binding_id, row in sorted(r2_bindings.items()):
        if binding_id in execution_ids:
            binding_class = "EXECUTION_CODE"
            source = "EXECUTION_CODE_COMMIT"
        elif binding_id in release_ids:
            binding_class = "LOCK_RELEASE_EVIDENCE"
            source = "R2_LOCK_RELEASE_COMMIT"
        elif binding_id in environment_ids:
            binding_class = "ENVIRONMENT_OR_BINARY"
            source = "FROZEN_SHA256_AND_ENVIRONMENT_CONTRACT"
        else:
            binding_class = "FROZEN_SCIENCE_OR_DATA"
            source = "INHERITED_LOCK_SHA256"
        records.append({
            "binding_id": binding_id,
            "repository_relative_path": row["repository_relative_path"],
            "sha256": row["sha256"],
            "bytes": int(row["bytes"]),
            "true_binding_class": binding_class,
            "true_verification_source": source,
            "old_name_heuristic_selected_execution_commit":
                binding_id.startswith("execution_"),
            "misclassified_by_old_name_heuristic":
                binding_id == "execution_control_patch_report",
        })

    verifier_path = (
        "experiments/mid360_formal_batch1/authorization/"
        "formal_registration_authorization_verify.py"
    )
    old_source = _git_show(root, BUG_COMMIT, verifier_path)
    findings = []
    for line_number, text in enumerate(old_source.splitlines(), 1):
        if "startswith(" in text or '"execution" in' in text:
            findings.append({
                "commit": BUG_COMMIT,
                "path": verifier_path,
                "line": line_number,
                "text": text.strip(),
                "decision_source": "BINDING_IDENTIFIER_TEXT",
                "status": "DEFECT",
            })

    r3_classification = {
        "EXECUTION_CODE": sorted(EXECUTION_CODE_PATHS),
        "LOCK_RELEASE_EVIDENCE": sorted(RELEASE_EVIDENCE_PATHS),
        "ENVIRONMENT_OR_BINARY": sorted(ENVIRONMENT_PATHS),
        "FROZEN_SCIENCE_OR_DATA": sorted(
            set(r2_bindings) - set(REPLACED_R2_BINDING_IDS)
        ),
    }
    return {
        "schema": "mid360_fmb1_authorization_binding_provenance_defect_audit_v1",
        "status": "PASS_DEFECT_CONFIRMED_AND_R3_MODEL_ENUMERATED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "r2_execution_code_commit": BUG_COMMIT,
        "r2_lock_release_commit":
            "ce83e131dfb896b500818e6ff4cf3399f80d05be",
        "r3_execution_code_commit_role": "TO_BE_BOUND_BY_EXEC_R3_LOCK",
        "r2_lock_path": lock_path.relative_to(root).as_posix(),
        "r2_lock_sha256": _sha(lock_path),
        "r2_binding_count": len(records),
        "r2_bindings": records,
        "heuristic_findings": findings,
        "defect": {
            "code": "AUTHORIZATION_VERIFIER_BINDING_PROVENANCE_DEFECT",
            "root_cause": (
                "binding identifier prefix was used to select the provenance commit"
            ),
            "failing_binding_id": "execution_control_patch_report",
            "actual_provenance": "LOCK_RELEASE_EVIDENCE",
            "incorrectly_required_commit": BUG_COMMIT,
            "artifact_exists_only_after_execution_code_commit": True,
            "attempt_003_outcome": "BLOCKED_BEFORE_BACKEND_INVOCATION",
        },
        "attempt_003": {
            "audit_commit": ATTEMPT_003_AUDIT_COMMIT,
            "authorization_id": ATTEMPT_003_AUTHORIZATION_ID,
            "authorization_sha256": ATTEMPT_003_AUTHORIZATION_SHA256,
            "authorization_status": "ISSUED_BUT_NOT_VERIFIED",
            "entered_in_use": False,
            "consumed": False,
            "future_execution_status": "VOID_FOR_FUTURE_EXECUTION",
            "reusable": False,
            "actual_formal_trials": 0,
        },
        "r3_contract": {
            "name_or_path_inference_allowed": False,
            "prefix_based_provenance_inference": False,
            "unknown_or_missing_binding_class_fails_closed": True,
            "binding_fields": [
                "binding_id", "repository_relative_path", "sha256", "bytes",
                "binding_class", "verification_source", "commit_role",
            ],
            "classification": r3_classification,
            "classification_counts": {
                key: len(value) for key, value in r3_classification.items()
            },
        },
        "science_state": {
            "scientific_protocol_changed": False,
            "final_dataset_changed": False,
            "trial_plan_scientific_content_changed": False,
            "backend_parameters_changed": False,
            "actual_formal_trials": 0,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    counts = payload["r3_contract"]["classification_counts"]
    return "\n".join([
        "# Exec-R2 authorization binding provenance defect audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "The independent authorization verifier selected a commit from the "
        "binding identifier prefix. `execution_control_patch_report` is a "
        "lock-release report, but the old rule required it in the earlier "
        "execution-code commit. Attempt 003 therefore failed before any "
        "backend invocation.",
        "",
        "Exec-R3 replaces that heuristic with exact per-binding metadata: "
        "`binding_class`, `verification_source`, and `commit_role`. Names and "
        "paths are opaque identifiers. Unknown or incomplete provenance fails "
        "closed.",
        "",
        "## Enumerated R3 classes",
        "",
        *(f"- `{key}`: {value}" for key, value in counts.items()),
        "",
        "`execution_control_patch_report` is explicitly "
        "`LOCK_RELEASE_EVIDENCE` and is authenticated in the exact lock-release "
        "tree, not in the execution-code commit.",
        "",
        "Attempt 003 remains `VOID_FOR_FUTURE_EXECUTION`, reusable=false, "
        "with zero formal trials. Science, final data, the 360-row plan, and "
        "backend parameters are unchanged.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    output = args.output_dir or (
        root / "results/mid360_formal_batch1/zero_perturbation_v1_1_exec_r3_lock"
    )
    payload = build_audit(root)
    _write_once(output / "authorization_binding_provenance_defect_audit.json",
                _canonical_json(payload))
    _write_once(output / "authorization_binding_provenance_defect_audit.md",
                render_markdown(payload).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
