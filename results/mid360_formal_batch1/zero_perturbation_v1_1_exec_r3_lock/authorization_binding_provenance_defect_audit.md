# Exec-R2 authorization binding provenance defect audit

Status: `PASS_DEFECT_CONFIRMED_AND_R3_MODEL_ENUMERATED`

The independent authorization verifier selected a commit from the binding identifier prefix. `execution_control_patch_report` is a lock-release report, but the old rule required it in the earlier execution-code commit. Attempt 003 therefore failed before any backend invocation.

Exec-R3 replaces that heuristic with exact per-binding metadata: `binding_class`, `verification_source`, and `commit_role`. Names and paths are opaque identifiers. Unknown or incomplete provenance fails closed.

## Enumerated R3 classes

- `EXECUTION_CODE`: 25
- `LOCK_RELEASE_EVIDENCE`: 8
- `ENVIRONMENT_OR_BINARY`: 3
- `FROZEN_SCIENCE_OR_DATA`: 43

`execution_control_patch_report` is explicitly `LOCK_RELEASE_EVIDENCE` and is authenticated in the exact lock-release tree, not in the execution-code commit.

Attempt 003 remains `VOID_FOR_FUTURE_EXECUTION`, reusable=false, with zero formal trials. Science, final data, the 360-row plan, and backend parameters are unchanged.
