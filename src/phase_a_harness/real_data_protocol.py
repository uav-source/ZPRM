"""Design-only Real-data Validation protocol framework and empty templates."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ELIGIBILITY_REQUIREMENTS = (
    ("R01", "At least two independent public data sources", "dataset citations and immutable release identifiers"),
    ("R02", "Independent high-accuracy 6DoF reference", "reference-system specification and accuracy evidence"),
    ("R03", "Evaluated scan is not used to build its target map", "map-construction lineage"),
    ("R04", "Target map and evaluated scan have independent source acquisition", "scan/map source identifiers"),
    ("R05", "Time synchronization and extrinsics are auditable", "timestamps, calibration, and transform chain"),
    ("R06", "Each dataset has at least 50 corridor or weak-geometry snapshots and 50 geometry-rich snapshots", "frozen snapshot inventory"),
    ("R07", "Scene labels are frozen before registration error is viewed", "timestamped blinded labeling record"),
    ("R08", "Open3D and PCL share identical inputs", "per-backend input checksums"),
    ("R09", "Frozen Open3D and PCL parameters are retained", "parameter-lock SHA-256"),
    ("R10", "Reference trajectory, map, and interpolation uncertainty are recorded", "uncertainty budget"),
    ("R11", "Primary effect exceeds synthetic and real-reference uncertainty budgets", "preregistered uncertainty-normalized comparison"),
    ("R12", "Continuous errors are primary; an arbitrary success threshold is not the sole result", "analysis contract"),
    ("R13", "Common-association metrics use the frozen offline definition", "common-analyzer implementation SHA-256"),
    ("R14", "Scene intervals cannot be reselected after results are inspected", "immutable interval-selection manifest"),
)

SNAPSHOT_SELECTION_FIELDS = (
    "dataset_id",
    "snapshot_id",
    "scene_interval_id",
    "scene_label",
    "label_frozen_at_utc",
    "labeler_blinded_to_registration_error",
    "scan_source_id",
    "target_map_source_id",
    "scan_timestamp",
    "reference_pose_timestamp",
    "reference_pose_interpolation_method",
    "scan_checksum",
    "target_map_checksum",
    "reference_pose_checksum",
    "open3d_pcl_shared_input",
    "eligibility_status",
    "exclusion_reason_predeclared",
)

UNCERTAINTY_BUDGET_FIELDS = (
    "dataset_id",
    "snapshot_id_or_group",
    "reference_translation_uncertainty_m",
    "reference_rotation_uncertainty_rad",
    "time_sync_translation_uncertainty_m",
    "time_sync_rotation_uncertainty_rad",
    "extrinsic_translation_uncertainty_m",
    "extrinsic_rotation_uncertainty_rad",
    "map_translation_uncertainty_m",
    "map_rotation_uncertainty_rad",
    "interpolation_translation_uncertainty_m",
    "interpolation_rotation_uncertainty_rad",
    "synthetic_translation_uncertainty_budget_m",
    "synthetic_rotation_uncertainty_budget_rad",
    "combined_translation_uncertainty_m",
    "combined_rotation_uncertainty_rad",
    "uncertainty_combination_rule",
    "evidence_reference",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    compact = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_real_data_protocol_framework(
    *, scientific_survival_audit_pass: bool
) -> dict[str, Any]:
    """Build the framework in memory; no dataset access is performed."""

    if scientific_survival_audit_pass is not True:
        raise PermissionError("scientific survival did not authorize protocol design")
    checklist = [
        {
            "requirement_id": identifier,
            "requirement": requirement,
            "mandatory": True,
            "required_evidence": evidence,
            "status": "NOT_EVALUATED",
        }
        for identifier, requirement, evidence in ELIGIBILITY_REQUIREMENTS
    ]
    core = {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_DATASET_ELIGIBILITY_COMPLETE": False,
        "REAL_DATA_PROTOCOL_FRAMEWORK_READY": True,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "dataset_download_count": 0,
        "dataset_execution_count": 0,
        "eligibility_requirement_count": len(checklist),
        "minimum_independent_public_dataset_count": 2,
        "minimum_snapshot_count_per_dataset_and_scene_group": {
            "CORRIDOR_OR_WEAK_GEOMETRY": 50,
            "GEOMETRY_RICH": 50,
        },
        "primary_comparison": "CORRIDOR_OR_WEAK_GEOMETRY_vs_GEOMETRY_RICH",
        "primary_outputs": [
            "translation_displacement",
            "rotation_displacement",
            "cross_backend_ranking",
            "turnover_error_association",
            "uncertainty_normalized_effect",
        ],
        "registration_backends": ["open3d_point_to_plane", "pcl_point_to_plane"],
        "registration_parameters": "FROZEN_FROM_SYNTHETIC_DEVELOPMENT",
        "schema_version": "real_data_validation_protocol_framework_v1",
        "scientific_survival_audit_pass": True,
    }
    protocol = {**core, "protocol_payload_sha256": _sha(core)}
    return {
        "checklist": checklist,
        "protocol": protocol,
        "snapshot_selection_fields": list(SNAPSHOT_SELECTION_FIELDS),
        "uncertainty_budget_fields": list(UNCERTAINTY_BUDGET_FIELDS),
    }


def real_data_protocol_markdown(bundle: Mapping[str, Any]) -> str:
    protocol = bundle["protocol"]
    lines = [
        "# Real-data Validation Protocol Framework v1",
        "",
        "This deliverable is a design framework only. No dataset has been downloaded, selected, or executed.",
        "",
        "- `REAL_DATA_PROTOCOL_FRAMEWORK_READY = true`",
        "- `REAL_DATA_DATASET_ELIGIBILITY_COMPLETE = false`",
        "- `REAL_DATA_RUN_AUTHORIZED = false`",
        "",
        "## Formal comparison",
        "",
        "Corridor/weak geometry is compared with geometry-rich environments using continuous translation and rotation displacement, cross-backend scene ranking, offline turnover/error association, and an uncertainty-normalized effect.",
        "",
        "## Eligibility requirements",
        "",
        "| ID | Mandatory requirement | Required evidence |",
        "|---|---|---|",
    ]
    for row in bundle["checklist"]:
        lines.append(
            f"| {row['requirement_id']} | {row['requirement']} | {row['required_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen analysis boundaries",
            "",
            "Scene labels and intervals must be frozen before registration errors are viewed. Open3D and PCL share byte-identical inputs and retain frozen parameters. Common-association metrics retain the synthetic offline definition and are not treated as backend-internal correspondences or causal proof.",
            "",
            "The primary scene effect must exceed both the synthetic uncertainty budget and the independent real-reference uncertainty budget. An arbitrary binary success threshold cannot replace the continuous primary outcomes.",
            "",
            f"Protocol payload SHA-256: `{protocol['protocol_payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_real_data_protocol_framework(
    repository_root: str | Path,
    *,
    scientific_survival_audit_pass: bool,
) -> dict[str, Any]:
    """Write five framework files, refusing overwrite and any run authorization."""

    bundle = build_real_data_protocol_framework(
        scientific_survival_audit_pass=scientific_survival_audit_pass
    )
    repository = Path(repository_root).resolve()
    destination = repository / "protocols"
    destination.mkdir(parents=True, exist_ok=True)
    checklist_fields = (
        "requirement_id",
        "requirement",
        "mandatory",
        "required_evidence",
        "status",
    )
    files = {
        destination / "real_data_validation_protocol_framework_v1.json": _canonical_json_bytes(bundle["protocol"]),
        destination / "real_data_validation_protocol_framework_v1.md": (
            real_data_protocol_markdown(bundle) + "\n"
        ).encode("utf-8"),
        destination / "real_data_dataset_eligibility_checklist.csv": _csv_bytes(bundle["checklist"], checklist_fields),
        destination / "real_data_snapshot_selection_template.csv": _csv_bytes([], SNAPSHOT_SELECTION_FIELDS),
        destination / "real_data_uncertainty_budget_template.csv": _csv_bytes([], UNCERTAINTY_BUDGET_FIELDS),
    }
    existing = [path for path in files if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace real-data protocol output: {existing[0]}")
    for path, payload in files.items():
        with path.open("xb") as stream:
            stream.write(payload)
    return {
        "CONFIRMATORY_RUN_AUTHORIZED": False,
        "MEASUREMENT_PAPER_MAINLINE_AUTHORIZED": False,
        "REAL_DATA_DATASET_ELIGIBILITY_COMPLETE": False,
        "REAL_DATA_PROTOCOL_FRAMEWORK_READY": True,
        "REAL_DATA_RUN_AUTHORIZED": False,
        "dataset_download_count": 0,
        "dataset_execution_count": 0,
        "eligibility_requirement_count": 14,
        "file_count": 5,
        "files": {
            path.relative_to(repository).as_posix(): hashlib.sha256(payload).hexdigest()
            for path, payload in files.items()
        },
    }


__all__ = [
    "ELIGIBILITY_REQUIREMENTS",
    "SNAPSHOT_SELECTION_FIELDS",
    "UNCERTAINTY_BUDGET_FIELDS",
    "build_real_data_protocol_framework",
    "materialize_real_data_protocol_framework",
    "real_data_protocol_markdown",
]
