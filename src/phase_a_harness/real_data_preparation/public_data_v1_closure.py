"""Build the immutable closure of the failed public-data v1 screening.

This producer only reads the four historical failure bundles.  It verifies
their existing ``SHA256SUMS`` closures before writing a separate six-file
closure; it never writes below any historical v1 directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .io import (
    atomic_write_bytes,
    atomic_write_json,
    compact_sha256,
    csv_bytes,
    sha256_file,
)
from .manifest import sha256sums_bytes


class PublicDataV1ClosureError(RuntimeError):
    """Historical v1 evidence is unavailable, altered, or semantically invalid."""


HISTORICAL_HEAD = "d304ea7a2a201aff21b48e5039422b63b0eb6e99"

PROTOCOL_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "path": "protocols/real_data_validation_protocol_framework_v1.md",
        "sha256": "4755a90dfd42de1650d50caa0facc4d32a0dba7f837ce9d867d28f030edfc834",
    },
    {
        "path": "protocols/real_data_validation_protocol_framework_v1.json",
        "sha256": "3b4dc774051806728a060a93aa1220ac7a4fc82a6e59181da2390cfdcb7bd86e",
    },
    {
        "path": "protocols/real_data_dataset_eligibility_checklist.csv",
        "sha256": "7a3c0b492f3783ea84bdda1eb8ff7fbd41ab73b9f5f7823806e2e30544a19c44",
    },
)

SOURCE_BUNDLES: tuple[dict[str, Any], ...] = (
    {
        "bundle_id": "IILABS_GRANDTOUR_V1",
        "candidate_ids": ["IILABS", "GrandTour"],
        "path": "frozen_assets/real_data_validation_v1",
        "sha256sums_entry_count": 47,
        "total_file_count": 48,
        "root_manifest_path": "SHA256SUMS",
        "root_manifest_sha256": "4e5a32a84831e5dbf08e23bb7449fdd3936219effe231aae2cabb732732f8f13",
        "semantic_manifest_path": "frozen_manifest_v1.json",
        "semantic_manifest_sha256": "697e1283106d4b2b6ac2f80de61b25183cf7418bb9041bc74884feb6784df9cd",
        "no_registration_attestation_path": "NO_ICP_ATTESTATION.json",
        "no_registration_attestation_sha256": "2f81bdb3fbb4e312a9106cc0e90507058df7a5bf1340ea2fcca6392c6892a6a9",
    },
    {
        "bundle_id": "RTS_GT_STAGE1_FAILED_V1",
        "candidate_ids": ["RTS-GT"],
        "path": "frozen_assets/real_data_rts_gt_stage1_failed_v1",
        "sha256sums_entry_count": 15,
        "total_file_count": 16,
        "root_manifest_path": "SHA256SUMS",
        "root_manifest_sha256": "8565f5e5f59d7f2b057d5507f0945c9ceaf2f1040c632d5f59417abfc22ad3e0",
        "semantic_manifest_path": "rts_gt_stage1_eligibility.json",
        "semantic_manifest_sha256": "5655c192c78f341fa776c76c320d19e0dcd1fdcb3e1388116135e16985b3c4a8",
        "no_registration_attestation_path": "NO_ICP_ATTESTATION.json",
        "no_registration_attestation_sha256": "9e50027384ab81b8e7fe47cfc823be5bc62a63d1501365759188ca95b4afede0",
    },
    {
        "bundle_id": "CAVERS_STAGE1_V1",
        "candidate_ids": ["CAVERS"],
        "path": "frozen_assets/real_data_cavers_stage1_v1",
        "sha256sums_entry_count": 30,
        "total_file_count": 31,
        "root_manifest_path": "SHA256SUMS",
        "root_manifest_sha256": "00332af73a7aca45ba60a7e2c1bdeb76169c6dd9d43ffd318a0fbe03666ec896",
        "semantic_manifest_path": "cavers_stage1_manifest.json",
        "semantic_manifest_sha256": "bbb92e78382969fbb65b5ae5ff176abb004137045c39d2da6fd1d55eb98fd5c8",
        "no_registration_attestation_path": "NO_ICP_ATTESTATION.json",
        "no_registration_attestation_sha256": "7447c4c68b3a434adf328592352688b9e3704ca0cbb2ac45579015c9b7004db1",
    },
    {
        "bundle_id": "BOREAS_STAGE1_V1",
        "candidate_ids": ["Boreas v1"],
        "path": "frozen_assets/real_data_boreas_stage1_v1",
        "sha256sums_entry_count": 28,
        "total_file_count": 29,
        "root_manifest_path": "SHA256SUMS",
        "root_manifest_sha256": "69a3a138617a315a3853dd33a519f9f142bd5bd9d08a3ace65b1dca9ba8d1975",
        "semantic_manifest_path": "boreas_stage1_manifest.json",
        "semantic_manifest_sha256": "29c582361eb83392e54d34ea4465ae4fb6b1813c89a33a5fa1eb85e1abbab112",
        "no_registration_attestation_path": "NO_ICP_ATTESTATION.json",
        "no_registration_attestation_sha256": "aa971b6956599a465d3b23985fa27cb65c00da682dc698eaaa530bbef7173b7c",
    },
)

CANDIDATE_FIELDS = (
    "candidate_id",
    "source_bundle",
    "historical_status",
    "screening_status",
    "failure_reason",
    "registration_count",
    "icp_count",
    "excluded_before_registration_results",
    "eligible_under_public_data_v1",
)

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "candidate_id": "IILABS",
        "source_bundle": "IILABS_GRANDTOUR_V1",
        "historical_status": "R03=FAIL;R04=FAIL;R05=FAIL;R10=FAIL",
        "screening_status": "EXCLUDED_BEFORE_REAL_REGISTRATION",
        "failure_reason": "cross-acquisition fixed world frame not proven",
        "registration_count": "0",
        "icp_count": "0",
        "excluded_before_registration_results": "true",
        "eligible_under_public_data_v1": "false",
    },
    {
        "candidate_id": "GrandTour",
        "source_bundle": "IILABS_GRANDTOUR_V1",
        "historical_status": "R03=FAIL;R04=FAIL;R05=FAIL;R10=FAIL",
        "screening_status": "EXCLUDED_BEFORE_REAL_REGISTRATION",
        "failure_reason": "GT-only overlap insufficient",
        "registration_count": "0",
        "icp_count": "0",
        "excluded_before_registration_results": "true",
        "eligible_under_public_data_v1": "false",
    },
    {
        "candidate_id": "RTS-GT",
        "source_bundle": "RTS_GT_STAGE1_FAILED_V1",
        "historical_status": "R05=FAIL;R10=FAIL;single_dataset_preregistration_ready=false",
        "screening_status": "EXCLUDED_BEFORE_REAL_REGISTRATION",
        "failure_reason": "LiDAR extrinsic/release lineage and uncertainty failure",
        "registration_count": "0",
        "icp_count": "0",
        "excluded_before_registration_results": "true",
        "eligible_under_public_data_v1": "false",
    },
    {
        "candidate_id": "CAVERS",
        "source_bundle": "CAVERS_STAGE1_V1",
        "historical_status": "R04=FAIL;R10=PARTIAL;CAVERS_STAGE1_READY=false",
        "screening_status": "EXCLUDED_BEFORE_REAL_REGISTRATION",
        "failure_reason": "cross-recording fixed world frame not proven",
        "registration_count": "0",
        "icp_count": "0",
        "excluded_before_registration_results": "true",
        "eligible_under_public_data_v1": "false",
    },
    {
        "candidate_id": "Boreas v1",
        "source_bundle": "BOREAS_STAGE1_V1",
        "historical_status": "R02=FAIL;R10=PARTIAL;BOREAS_STAGE1_READY=false",
        "screening_status": "EXCLUDED_BEFORE_REAL_REGISTRATION",
        "failure_reason": "R02 failed under the stricter Stage-1 independence interpretation; R10 partial",
        "registration_count": "0",
        "icp_count": "0",
        "excluded_before_registration_results": "true",
        "eligible_under_public_data_v1": "false",
    },
)

CLOSURE_FILES = frozenset(
    {
        "public_data_v1_closure_summary.md",
        "public_data_v1_closure.json",
        "public_data_v1_candidate_screening.csv",
        "public_data_v1_evidence_manifest.json",
        "NO_REAL_REGISTRATION_ATTESTATION.json",
        "SHA256SUMS",
    }
)

_ZERO_ATTESTATION_FIELDS = (
    "open3d_registration_call_count",
    "pcl_cli_invocation_count",
    "other_registration_process_count",
    "real_trial_result_count",
    "registration_execution_count",
)


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PublicDataV1ClosureError(f"unsafe SHA256SUMS path: {value}")
    return relative


def _parse_and_verify_sha256sums(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise PublicDataV1ClosureError(f"missing or unsafe SHA256SUMS: {root}")
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise PublicDataV1ClosureError(f"malformed SHA256SUMS row in {root}: {line!r}")
        digest, name = match.groups()
        relative = _safe_relative_path(name)
        if name in rows or name == "SHA256SUMS":
            raise PublicDataV1ClosureError(f"duplicate or recursive SHA256SUMS path: {name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise PublicDataV1ClosureError(f"missing or unsafe source artifact: {path}")
        resolved = path.resolve(strict=True)
        if root.resolve(strict=True) not in resolved.parents:
            raise PublicDataV1ClosureError(f"source artifact escapes bundle: {path}")
        if sha256_file(path) != digest:
            raise PublicDataV1ClosureError(f"source SHA256 mismatch: {path}")
        rows[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(rows) | {"SHA256SUMS"}
    if actual != expected:
        raise PublicDataV1ClosureError(
            f"source bundle closure mismatch at {root}: "
            f"missing={sorted(expected - actual)}, orphan={sorted(actual - expected)}"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise PublicDataV1ClosureError(f"source bundle contains a symlink: {root}")
    return rows


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicDataV1ClosureError(f"invalid JSON evidence: {path}") from error


def _require_zero_attestation(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("pass") is not True:
        raise PublicDataV1ClosureError(f"historical no-registration attestation is not passing: {path}")
    for field in _ZERO_ATTESTATION_FIELDS:
        if value.get(field) != 0:
            raise PublicDataV1ClosureError(f"historical registration count is nonzero: {path}:{field}")
    for optional in ("estimated_transform_count", "estimated_transform_file_count"):
        if optional in value and value[optional] != 0:
            raise PublicDataV1ClosureError(f"historical transform count is nonzero: {path}:{optional}")
    return value


def _verify_historical_semantics(repository: Path) -> None:
    shared = repository / "frozen_assets/real_data_validation_v1"
    summary = _load_json(shared / "preparation_summary.json")
    if any(
        summary.get(field) != 0
        for field in (
            "ICP_EXECUTION_COUNT",
            "actual_icp_execution_count",
            "actual_registration_execution_count",
            "registration_execution_count",
        )
    ) or summary.get("final_conclusion") != "PREREGISTRATION_NOT_READY":
        raise PublicDataV1ClosureError("IILABS/GrandTour historical failure state changed")
    requirements = {
        row.get("requirement_id"): row.get("global_status")
        for row in _load_json(shared / "r01_r10_eligibility_audit.json").get("requirements", [])
    }
    if requirements != {
        "R01": "PASS",
        "R02": "PASS",
        "R03": "FAIL",
        "R04": "FAIL",
        "R05": "FAIL",
        "R06": "BLOCKED",
        "R07": "BLOCKED",
        "R08": "BLOCKED",
        "R09": "PASS",
        "R10": "FAIL",
    }:
        raise PublicDataV1ClosureError("IILABS/GrandTour eligibility state changed")
    world = _load_json(shared / "iilabs/common_world_frame_audit.json")
    if world.get("same_fixed_world_frame_proven") is not False or world.get("status") != "FAIL":
        raise PublicDataV1ClosureError("IILABS world-frame failure changed")
    overlap = _load_json(shared / "grandtour/gt_overlap_report.json").get("gt_only_overlap", {})
    if set(overlap) != {"SPX-1->SPX-3", "SPX-3->SPX-1"} or any(
        row.get("eligibility_status") != "FAIL" for row in overlap.values()
    ):
        raise PublicDataV1ClosureError("GrandTour GT-only overlap failure changed")
    blind = _load_json(shared / "blinded_labeling_record.json")
    if (
        blind.get("real_registration_result_file_count") != 0
        or blind.get("registration_derived_field_access_count") != 0
    ):
        raise PublicDataV1ClosureError("IILABS/GrandTour exclusion was not before results")

    rts = _load_json(
        repository
        / "frozen_assets/real_data_rts_gt_stage1_failed_v1/rts_gt_stage1_eligibility.json"
    )
    if (
        rts.get("R05") != "FAIL"
        or rts.get("R10") != "FAIL"
        or rts.get("single_dataset_preregistration_ready") is not False
        or rts.get("stage1_status") != "FAIL_STOPPED_BEFORE_LIDAR_DOWNLOAD"
        or rts.get("actual_trials") != 0
    ):
        raise PublicDataV1ClosureError("RTS-GT historical failure state changed")

    cavers = _load_json(
        repository / "frozen_assets/real_data_cavers_stage1_v1/cavers_stage1_eligibility.json"
    )
    if (
        cavers.get("R04") != "FAIL"
        or cavers.get("R10") != "PARTIAL"
        or cavers.get("CAVERS_STAGE1_READY") is not False
        or cavers.get("registration_execution_count") != 0
    ):
        raise PublicDataV1ClosureError("CAVERS historical failure state changed")

    boreas = _load_json(
        repository / "frozen_assets/real_data_boreas_stage1_v1/boreas_stage1_eligibility.json"
    )
    if (
        boreas.get("R02") != "FAIL"
        or boreas.get("R10") != "PARTIAL"
        or boreas.get("BOREAS_STAGE1_READY") is not False
        or boreas.get("registration_execution_count") != 0
    ):
        raise PublicDataV1ClosureError("Boreas v1 historical R02/R10 conclusion changed")


def inspect_public_data_v1_sources(repository: str | Path) -> dict[str, Any]:
    """Authenticate the v1 protocols and all four historical bundle closures."""

    repository_input = Path(repository)
    if repository_input.is_symlink():
        raise PublicDataV1ClosureError("repository root is missing or a symlink")
    repository_path = repository_input.resolve(strict=True)
    if not repository_path.is_dir():
        raise PublicDataV1ClosureError("repository root is missing or a symlink")

    protocol_rows: list[dict[str, Any]] = []
    for binding in PROTOCOL_BINDINGS:
        path = repository_path / binding["path"]
        if path.is_symlink() or not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise PublicDataV1ClosureError(f"v1 protocol hash mismatch: {binding['path']}")
        protocol_rows.append(
            {**binding, "size_bytes": path.stat().st_size}
        )

    bundle_rows: list[dict[str, Any]] = []
    for spec in SOURCE_BUNDLES:
        root = repository_path / spec["path"]
        if root.is_symlink() or not root.is_dir():
            raise PublicDataV1ClosureError(f"historical source bundle is absent or unsafe: {root}")
        sums = _parse_and_verify_sha256sums(root)
        if len(sums) != spec["sha256sums_entry_count"]:
            raise PublicDataV1ClosureError(f"source entry count changed: {spec['bundle_id']}")
        if len(sums) + 1 != spec["total_file_count"]:
            raise PublicDataV1ClosureError(f"source total file count changed: {spec['bundle_id']}")
        if sha256_file(root / spec["root_manifest_path"]) != spec["root_manifest_sha256"]:
            raise PublicDataV1ClosureError(f"source root manifest changed: {spec['bundle_id']}")
        if sha256_file(root / spec["semantic_manifest_path"]) != spec["semantic_manifest_sha256"]:
            raise PublicDataV1ClosureError(f"source semantic manifest changed: {spec['bundle_id']}")
        if (
            sha256_file(root / spec["no_registration_attestation_path"])
            != spec["no_registration_attestation_sha256"]
        ):
            raise PublicDataV1ClosureError(f"source no-registration attestation changed: {spec['bundle_id']}")
        _require_zero_attestation(root / spec["no_registration_attestation_path"])
        bundle_rows.append(dict(spec))

    _verify_historical_semantics(repository_path)
    return {
        "historical_head": HISTORICAL_HEAD,
        "protocols": protocol_rows,
        "source_bundles": bundle_rows,
        "source_bundle_count": len(bundle_rows),
        "verified_sha256sums_entry_count": sum(
            row["sha256sums_entry_count"] for row in bundle_rows
        ),
        "verified_source_total_file_count": sum(row["total_file_count"] for row in bundle_rows),
    }


def _evidence_manifest(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate_count": len(CANDIDATES),
        "historical_head": source["historical_head"],
        "manifest_kind": "PUBLIC_DATA_V1_SCREENING_EVIDENCE_CLOSURE",
        "protocols": source["protocols"],
        "schema_version": "public_data_v1_evidence_manifest_v1",
        "source_bundle_count": source["source_bundle_count"],
        "source_bundles": source["source_bundles"],
        "verified_sha256sums_entry_count": source["verified_sha256sums_entry_count"],
        "verified_source_total_file_count": source["verified_source_total_file_count"],
    }
    return {**payload, "manifest_root_sha256": compact_sha256(payload)}


def _no_registration_attestation(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "bundle_id": bundle["bundle_id"],
            "candidate_ids": bundle["candidate_ids"],
            "no_registration_attestation_path": (
                f"{bundle['path']}/{bundle['no_registration_attestation_path']}"
            ),
            "no_registration_attestation_sha256": bundle[
                "no_registration_attestation_sha256"
            ],
            "open3d_registration_call_count": 0,
            "other_registration_process_count": 0,
            "pcl_cli_invocation_count": 0,
            "real_trial_result_count": 0,
            "registration_execution_count": 0,
        }
        for bundle in source["source_bundles"]
    ]
    return {
        "PUBLIC_DATA_V1_ICP_COUNT": 0,
        "PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT": 0,
        "all_candidates_excluded_before_registration_results": True,
        "candidate_count": len(CANDIDATES),
        "estimated_transform_count": 0,
        "evidence_bundle_attestations": rows,
        "open3d_registration_call_count": 0,
        "other_registration_process_count": 0,
        "pass": True,
        "pcl_cli_invocation_count": 0,
        "real_trial_result_count": 0,
        "registration_execution_count": 0,
        "schema_version": "no_real_registration_attestation_v1",
    }


def _summary_markdown(source: Mapping[str, Any]) -> bytes:
    rows = [
        "# Public-data v1 screening closure",
        "",
        "Public-data v1 is closed as a failed eligibility screening, not as a successful real-data qualification.",
        "All five candidates were excluded before any real registration result existed.",
        "",
        "| Candidate | Historical status | Frozen failure reason |",
        "|---|---|---|",
    ]
    rows.extend(
        f"| {row['candidate_id']} | `{row['historical_status']}` | {row['failure_reason']} |"
        for row in CANDIDATES
    )
    rows.extend(
        [
            "",
            "Boreas v1 remains `R02=FAIL`, `R10=PARTIAL`, and `BOREAS_STAGE1_READY=false`; this closure does not reinterpret it.",
            "",
            "```text",
            "PUBLIC_DATA_V1_SCREENING_COMPLETE=true",
            "PUBLIC_DATA_V1_CANDIDATE_COUNT=5",
            "PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT=0",
            "PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT=0",
            "PUBLIC_DATA_V1_ICP_COUNT=0",
            "PUBLIC_DATA_V1_RUN_AUTHORIZED=false",
            "PUBLIC_DATA_V1_CLOSED=true",
            "```",
            "",
            f"Verified source bundles: {source['source_bundle_count']}; source files including their root SHA256SUMS: {source['verified_source_total_file_count']}.",
            "",
        ]
    )
    return "\n".join(rows).encode("utf-8")


def build_public_data_v1_closure(
    *,
    repository: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and return the fixed v1 closure after authenticating all inputs."""

    repository_input = Path(repository)
    if repository_input.is_symlink():
        raise PublicDataV1ClosureError("repository root is missing or a symlink")
    repository_path = repository_input.resolve(strict=True)
    output = Path(output_root)
    if not output.is_absolute():
        raise PublicDataV1ClosureError(f"output root is not absolute: {output}")
    if output.exists() and output.is_symlink():
        raise PublicDataV1ClosureError(f"output root is a symlink: {output}")
    resolved_output = output.resolve(strict=False)
    for spec in SOURCE_BUNDLES:
        source_root = (repository_path / spec["path"]).resolve(strict=True)
        if resolved_output == source_root or source_root in resolved_output.parents:
            raise PublicDataV1ClosureError("closure output may not overlap a historical v1 bundle")

    source = inspect_public_data_v1_sources(repository_path)
    output.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in output.iterdir()):
        raise PublicDataV1ClosureError("closure output contains a symlink")
    unexpected = {path.name for path in output.iterdir()} - CLOSURE_FILES
    if unexpected:
        raise PublicDataV1ClosureError(f"closure output contains unexpected entries: {sorted(unexpected)}")

    candidate_bytes = csv_bytes(CANDIDATES, CANDIDATE_FIELDS)
    evidence = _evidence_manifest(source)
    attestation = _no_registration_attestation(source)
    summary_bytes = _summary_markdown(source)

    atomic_write_bytes(
        output / "public_data_v1_candidate_screening.csv",
        candidate_bytes,
        overwrite=overwrite,
    )
    atomic_write_json(
        output / "public_data_v1_evidence_manifest.json",
        evidence,
        overwrite=overwrite,
    )
    atomic_write_json(
        output / "NO_REAL_REGISTRATION_ATTESTATION.json",
        attestation,
        overwrite=overwrite,
    )
    atomic_write_bytes(
        output / "public_data_v1_closure_summary.md",
        summary_bytes,
        overwrite=overwrite,
    )

    closure = {
        "PUBLIC_DATA_V1_CANDIDATE_COUNT": 5,
        "PUBLIC_DATA_V1_CLOSED": True,
        "PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT": 0,
        "PUBLIC_DATA_V1_ICP_COUNT": 0,
        "PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT": 0,
        "PUBLIC_DATA_V1_RUN_AUTHORIZED": False,
        "PUBLIC_DATA_V1_SCREENING_COMPLETE": True,
        "all_candidates_excluded_before_registration_results": True,
        "boreas_v1_conclusion_preserved": True,
        "candidate_screening_sha256": sha256_file(
            output / "public_data_v1_candidate_screening.csv"
        ),
        "evidence_manifest_root_sha256": evidence["manifest_root_sha256"],
        "evidence_manifest_sha256": sha256_file(
            output / "public_data_v1_evidence_manifest.json"
        ),
        "historical_protocol_v1_preserved": True,
        "no_real_registration_attestation_sha256": sha256_file(
            output / "NO_REAL_REGISTRATION_ATTESTATION.json"
        ),
        "schema_version": "public_data_v1_closure_v1",
        "source_bundle_count": 4,
    }
    atomic_write_json(output / "public_data_v1_closure.json", closure, overwrite=overwrite)

    checksum_members = sorted(CLOSURE_FILES - {"SHA256SUMS"})
    atomic_write_bytes(
        output / "SHA256SUMS",
        sha256sums_bytes(output, checksum_members),
        overwrite=overwrite,
    )
    return closure


__all__ = [
    "CANDIDATES",
    "CANDIDATE_FIELDS",
    "CLOSURE_FILES",
    "PublicDataV1ClosureError",
    "build_public_data_v1_closure",
    "inspect_public_data_v1_sources",
]
