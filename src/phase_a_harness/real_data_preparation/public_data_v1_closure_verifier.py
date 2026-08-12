"""Independent verifier for the immutable public-data v1 closure.

This module intentionally does not import the closure producer.  Historical
protocol and bundle roots, candidate decisions, and closure semantics are
redeclared here so a producer-side change cannot silently redefine success.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .io import canonical_json_bytes, compact_sha256, sha256_file


class PublicDataV1ClosureVerificationError(RuntimeError):
    """The historical evidence or closure is incomplete, altered, or misleading."""


EXPECTED_HISTORICAL_HEAD = "d304ea7a2a201aff21b48e5039422b63b0eb6e99"

EXPECTED_PROTOCOLS: tuple[dict[str, Any], ...] = (
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

EXPECTED_SOURCE_BUNDLES: tuple[dict[str, Any], ...] = (
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

EXPECTED_CANDIDATE_FIELDS = (
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

EXPECTED_CANDIDATES: tuple[dict[str, str], ...] = (
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

EXPECTED_CLOSURE_FILES = frozenset(
    {
        "public_data_v1_closure_summary.md",
        "public_data_v1_closure.json",
        "public_data_v1_candidate_screening.csv",
        "public_data_v1_evidence_manifest.json",
        "NO_REAL_REGISTRATION_ATTESTATION.json",
        "SHA256SUMS",
    }
)

ZERO_REGISTRATION_FIELDS = (
    "open3d_registration_call_count",
    "pcl_cli_invocation_count",
    "other_registration_process_count",
    "real_trial_result_count",
    "registration_execution_count",
)


def _fail(message: str) -> None:
    raise PublicDataV1ClosureVerificationError(message)


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: {actual!r} != {expected!r}")


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail(f"unsafe manifest path: {value}")
    return relative


def _load_canonical_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicDataV1ClosureVerificationError(f"invalid JSON: {path}") from error
    if canonical_json_bytes(value) != path.read_bytes():
        _fail(f"JSON is not canonical: {path}")
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicDataV1ClosureVerificationError(f"invalid historical JSON: {path}") from error


def _verify_source_sums(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        _fail(f"historical bundle is missing or a symlink: {root}")
    checksum = root / "SHA256SUMS"
    if checksum.is_symlink() or not checksum.is_file():
        _fail(f"historical SHA256SUMS is missing or unsafe: {root}")
    rows: dict[str, str] = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _fail(f"malformed historical SHA256SUMS row: {line!r}")
        digest, name = match.groups()
        relative = _safe_relative(name)
        if name in rows or name == "SHA256SUMS":
            _fail(f"duplicate or recursive historical SHA256SUMS path: {name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            _fail(f"historical source artifact is missing or unsafe: {path}")
        resolved = path.resolve(strict=True)
        if root.resolve(strict=True) not in resolved.parents:
            _fail(f"historical source artifact escapes its bundle: {path}")
        _same(sha256_file(path), digest, f"historical SHA256 {path}")
        rows[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(rows) | {"SHA256SUMS"}
    if actual != expected:
        _fail(
            f"historical source closure mismatch at {root}: "
            f"missing={sorted(expected - actual)}, orphan={sorted(actual - expected)}"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        _fail(f"historical source bundle contains a symlink: {root}")
    return rows


def _verify_no_registration(path: Path) -> None:
    value = _load_json(path)
    _same(value.get("pass"), True, f"no-registration pass {path}")
    for field in ZERO_REGISTRATION_FIELDS:
        _same(value.get(field), 0, f"zero registration field {path}:{field}")
    for field in ("estimated_transform_count", "estimated_transform_file_count"):
        if field in value:
            _same(value[field], 0, f"zero transform field {path}:{field}")


def _verify_historical_meaning(repository: Path) -> None:
    shared = repository / "frozen_assets/real_data_validation_v1"
    summary = _load_json(shared / "preparation_summary.json")
    _same(summary.get("final_conclusion"), "PREREGISTRATION_NOT_READY", "shared final status")
    for field in (
        "ICP_EXECUTION_COUNT",
        "actual_icp_execution_count",
        "actual_registration_execution_count",
        "registration_execution_count",
    ):
        _same(summary.get(field), 0, f"shared zero count {field}")
    requirements = {
        row.get("requirement_id"): row.get("global_status")
        for row in _load_json(shared / "r01_r10_eligibility_audit.json").get("requirements", [])
    }
    _same(
        requirements,
        {
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
        },
        "shared v1 eligibility",
    )
    iilabs = _load_json(shared / "iilabs/common_world_frame_audit.json")
    _same(iilabs.get("status"), "FAIL", "IILABS world-frame status")
    _same(iilabs.get("same_fixed_world_frame_proven"), False, "IILABS fixed-world proof")
    grandtour = _load_json(shared / "grandtour/gt_overlap_report.json").get("gt_only_overlap", {})
    _same(set(grandtour), {"SPX-1->SPX-3", "SPX-3->SPX-1"}, "GrandTour pairs")
    if any(row.get("eligibility_status") != "FAIL" for row in grandtour.values()):
        _fail("GrandTour historical GT-only overlap failure changed")
    blind = _load_json(shared / "blinded_labeling_record.json")
    _same(blind.get("real_registration_result_file_count"), 0, "shared result-file count")
    _same(blind.get("registration_derived_field_access_count"), 0, "shared result access count")

    rts = _load_json(
        repository
        / "frozen_assets/real_data_rts_gt_stage1_failed_v1/rts_gt_stage1_eligibility.json"
    )
    for key, expected in (
        ("R05", "FAIL"),
        ("R10", "FAIL"),
        ("single_dataset_preregistration_ready", False),
        ("stage1_status", "FAIL_STOPPED_BEFORE_LIDAR_DOWNLOAD"),
        ("actual_trials", 0),
    ):
        _same(rts.get(key), expected, f"RTS-GT {key}")

    cavers = _load_json(
        repository / "frozen_assets/real_data_cavers_stage1_v1/cavers_stage1_eligibility.json"
    )
    for key, expected in (
        ("R04", "FAIL"),
        ("R10", "PARTIAL"),
        ("CAVERS_STAGE1_READY", False),
        ("registration_execution_count", 0),
    ):
        _same(cavers.get(key), expected, f"CAVERS {key}")

    boreas = _load_json(
        repository / "frozen_assets/real_data_boreas_stage1_v1/boreas_stage1_eligibility.json"
    )
    for key, expected in (
        ("R02", "FAIL"),
        ("R10", "PARTIAL"),
        ("BOREAS_STAGE1_READY", False),
        ("registration_execution_count", 0),
    ):
        _same(boreas.get(key), expected, f"Boreas v1 {key}")


def _verify_historical_sources(repository: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocols: list[dict[str, Any]] = []
    for expected in EXPECTED_PROTOCOLS:
        path = repository / expected["path"]
        if path.is_symlink() or not path.is_file():
            _fail(f"v1 protocol is missing or unsafe: {expected['path']}")
        _same(sha256_file(path), expected["sha256"], f"v1 protocol SHA {expected['path']}")
        protocols.append({**expected, "size_bytes": path.stat().st_size})

    bundles: list[dict[str, Any]] = []
    for expected in EXPECTED_SOURCE_BUNDLES:
        root = repository / expected["path"]
        rows = _verify_source_sums(root)
        _same(len(rows), expected["sha256sums_entry_count"], f"source entry count {expected['bundle_id']}")
        _same(len(rows) + 1, expected["total_file_count"], f"source file count {expected['bundle_id']}")
        _same(
            sha256_file(root / expected["root_manifest_path"]),
            expected["root_manifest_sha256"],
            f"source root manifest {expected['bundle_id']}",
        )
        _same(
            sha256_file(root / expected["semantic_manifest_path"]),
            expected["semantic_manifest_sha256"],
            f"source semantic manifest {expected['bundle_id']}",
        )
        attestation = root / expected["no_registration_attestation_path"]
        _same(
            sha256_file(attestation),
            expected["no_registration_attestation_sha256"],
            f"source no-registration attestation {expected['bundle_id']}",
        )
        _verify_no_registration(attestation)
        bundles.append(dict(expected))
    _verify_historical_meaning(repository)
    return protocols, bundles


def _verify_closure_sha256sums(root: Path) -> dict[str, str]:
    entries = {path.name for path in root.iterdir()}
    if entries != EXPECTED_CLOSURE_FILES or any(
        not path.is_file() or path.is_symlink() for path in root.iterdir()
    ):
        _fail(
            "closure file set mismatch: "
            f"missing={sorted(EXPECTED_CLOSURE_FILES - entries)}, "
            f"extra={sorted(entries - EXPECTED_CLOSURE_FILES)}"
        )
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            _fail(f"malformed closure SHA256SUMS row: {line!r}")
        digest, name = match.groups()
        if name in rows or name == "SHA256SUMS":
            _fail(f"duplicate or recursive closure SHA256SUMS path: {name}")
        rows[name] = digest
    _same(set(rows), EXPECTED_CLOSURE_FILES - {"SHA256SUMS"}, "closure SHA256SUMS inventory")
    for name, digest in rows.items():
        _same(sha256_file(root / name), digest, f"closure SHA256 {name}")
    return rows


def _csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _same(reader.fieldnames, list(fields), f"CSV header {path.name}")
        return list(reader)


def _expected_attestation(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "PUBLIC_DATA_V1_ICP_COUNT": 0,
        "PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT": 0,
        "all_candidates_excluded_before_registration_results": True,
        "candidate_count": 5,
        "estimated_transform_count": 0,
        "evidence_bundle_attestations": [
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
            for bundle in bundles
        ],
        "open3d_registration_call_count": 0,
        "other_registration_process_count": 0,
        "pass": True,
        "pcl_cli_invocation_count": 0,
        "real_trial_result_count": 0,
        "registration_execution_count": 0,
        "schema_version": "no_real_registration_attestation_v1",
    }


def _expected_summary() -> str:
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
        for row in EXPECTED_CANDIDATES
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
            "Verified source bundles: 4; source files including their root SHA256SUMS: 124.",
            "",
        ]
    )
    return "\n".join(rows)


def verify_public_data_v1_closure(
    *,
    repository: str | Path,
    closure_root: str | Path,
) -> dict[str, Any]:
    """Independently authenticate source history and the six-file v1 closure."""

    repository_input = Path(repository)
    root_input = Path(closure_root)
    if repository_input.is_symlink():
        _fail("repository root is missing or a symlink")
    if root_input.is_symlink():
        _fail("closure root is missing or a symlink")
    repository_path = repository_input.resolve(strict=True)
    root = root_input.resolve(strict=True)
    if not repository_path.is_dir():
        _fail("repository root is missing or a symlink")
    if not root.is_dir():
        _fail("closure root is missing or a symlink")

    protocols, bundles = _verify_historical_sources(repository_path)
    _verify_closure_sha256sums(root)

    candidate_rows = _csv_rows(
        root / "public_data_v1_candidate_screening.csv", EXPECTED_CANDIDATE_FIELDS
    )
    _same(candidate_rows, list(EXPECTED_CANDIDATES), "candidate screening rows")

    evidence = _load_canonical_json(root / "public_data_v1_evidence_manifest.json")
    unsigned_evidence = dict(evidence)
    evidence_root = unsigned_evidence.pop("manifest_root_sha256", None)
    _same(compact_sha256(unsigned_evidence), evidence_root, "evidence manifest root SHA")
    expected_evidence = {
        "candidate_count": 5,
        "historical_head": EXPECTED_HISTORICAL_HEAD,
        "manifest_kind": "PUBLIC_DATA_V1_SCREENING_EVIDENCE_CLOSURE",
        "protocols": protocols,
        "schema_version": "public_data_v1_evidence_manifest_v1",
        "source_bundle_count": 4,
        "source_bundles": bundles,
        "verified_sha256sums_entry_count": 120,
        "verified_source_total_file_count": 124,
    }
    _same(unsigned_evidence, expected_evidence, "evidence manifest payload")

    attestation = _load_canonical_json(root / "NO_REAL_REGISTRATION_ATTESTATION.json")
    _same(attestation, _expected_attestation(bundles), "closure no-registration attestation")

    closure = _load_canonical_json(root / "public_data_v1_closure.json")
    expected_closure = {
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
            root / "public_data_v1_candidate_screening.csv"
        ),
        "evidence_manifest_root_sha256": evidence_root,
        "evidence_manifest_sha256": sha256_file(
            root / "public_data_v1_evidence_manifest.json"
        ),
        "historical_protocol_v1_preserved": True,
        "no_real_registration_attestation_sha256": sha256_file(
            root / "NO_REAL_REGISTRATION_ATTESTATION.json"
        ),
        "schema_version": "public_data_v1_closure_v1",
        "source_bundle_count": 4,
    }
    _same(closure, expected_closure, "closure decision")

    summary = (root / "public_data_v1_closure_summary.md").read_text(encoding="utf-8")
    _same(summary, _expected_summary(), "closure summary")

    return {
        "PUBLIC_DATA_V1_CANDIDATE_COUNT": 5,
        "PUBLIC_DATA_V1_CLOSED": True,
        "PUBLIC_DATA_V1_CLOSURE_VERIFICATION_PASS": True,
        "PUBLIC_DATA_V1_ELIGIBLE_DATASET_COUNT": 0,
        "PUBLIC_DATA_V1_ICP_COUNT": 0,
        "PUBLIC_DATA_V1_REAL_REGISTRATION_COUNT": 0,
        "PUBLIC_DATA_V1_RUN_AUTHORIZED": False,
        "PUBLIC_DATA_V1_SCREENING_COMPLETE": True,
        "closure_file_count": 6,
        "historical_protocol_file_count": 3,
        "source_bundle_count": 4,
        "verified_sha256sums_entry_count": 120,
        "verified_source_total_file_count": 124,
        "verification_pass": True,
    }


__all__ = [
    "PublicDataV1ClosureVerificationError",
    "verify_public_data_v1_closure",
]
