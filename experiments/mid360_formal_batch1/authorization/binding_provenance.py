"""Explicit provenance contract for FMB1 Exec-R3 lock bindings.

Binding identifiers are opaque labels.  Verification routing is controlled
only by the three explicit provenance fields defined here; names and paths
never select a commit or verification policy.
"""

from __future__ import annotations

from typing import Any, Mapping


EXECUTION_CODE = "EXECUTION_CODE"
LOCK_RELEASE_EVIDENCE = "LOCK_RELEASE_EVIDENCE"
FROZEN_SCIENCE_OR_DATA = "FROZEN_SCIENCE_OR_DATA"
ENVIRONMENT_OR_BINARY = "ENVIRONMENT_OR_BINARY"

GIT_BLOB_AT_COMMIT = "GIT_BLOB_AT_COMMIT"
INHERITED_LOCK_SHA256 = "INHERITED_EXEC_R2_LOCK_SHA256"
FROZEN_ENVIRONMENT_CONTRACT = "FROZEN_SHA256_AND_ENVIRONMENT_CONTRACT"

EXECUTION_CODE_COMMIT = "EXECUTION_CODE_COMMIT"
LOCK_RELEASE_COMMIT = "LOCK_RELEASE_COMMIT"
NO_COMMIT = "NONE"

BINDING_KEYS = {
    "binding_id",
    "repository_relative_path",
    "sha256",
    "bytes",
    "binding_class",
    "verification_source",
    "commit_role",
}

PROVENANCE_CONTRACT = {
    EXECUTION_CODE: (GIT_BLOB_AT_COMMIT, EXECUTION_CODE_COMMIT),
    LOCK_RELEASE_EVIDENCE: (GIT_BLOB_AT_COMMIT, LOCK_RELEASE_COMMIT),
    FROZEN_SCIENCE_OR_DATA: (INHERITED_LOCK_SHA256, NO_COMMIT),
    ENVIRONMENT_OR_BINARY: (FROZEN_ENVIRONMENT_CONTRACT, NO_COMMIT),
}


class BindingProvenanceError(ValueError):
    pass


def validate_binding_metadata(binding_id: str, row: Mapping[str, Any]) -> None:
    """Validate metadata shape without performing source verification."""

    if set(row) != BINDING_KEYS:
        raise BindingProvenanceError(
            f"binding metadata keys differ for {binding_id}: "
            f"missing={sorted(BINDING_KEYS-set(row))}, "
            f"extra={sorted(set(row)-BINDING_KEYS)}"
        )
    if row.get("binding_id") != binding_id:
        raise BindingProvenanceError(f"binding_id differs for {binding_id}")
    binding_class = row.get("binding_class")
    if binding_class not in PROVENANCE_CONTRACT:
        raise BindingProvenanceError(f"unknown binding_class for {binding_id}")
    expected_source, expected_role = PROVENANCE_CONTRACT[str(binding_class)]
    if row.get("verification_source") != expected_source:
        raise BindingProvenanceError(
            f"verification_source differs for {binding_id}"
        )
    if row.get("commit_role") != expected_role:
        raise BindingProvenanceError(f"commit_role differs for {binding_id}")
    if not isinstance(row.get("repository_relative_path"), str):
        raise BindingProvenanceError(f"binding path is invalid for {binding_id}")
    if not isinstance(row.get("sha256"), str):
        raise BindingProvenanceError(f"binding SHA is invalid for {binding_id}")
    byte_count = row.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise BindingProvenanceError(f"binding byte count is invalid for {binding_id}")
