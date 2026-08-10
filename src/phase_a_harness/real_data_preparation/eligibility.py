"""Small fail-closed eligibility assertions shared by builders and tests."""

from __future__ import annotations


def mission_reference_eligible(
    *,
    public_file_present: bool,
    independent_non_lidar: bool,
    complete_6dof: bool,
    accuracy_evidence_present: bool,
    reference_status: str,
) -> bool:
    if reference_status == "HIDDEN_OR_UNAVAILABLE":
        return False
    return all(
        (public_file_present, independent_non_lidar, complete_6dof, accuracy_evidence_present)
    )


def assert_byte_identical_backend_bundle(open3d_sha256: str, pcl_sha256: str) -> None:
    if len(open3d_sha256) != 64 or len(pcl_sha256) != 64:
        raise ValueError("bundle SHA-256 must have 64 hex characters")
    try:
        int(open3d_sha256, 16)
        int(pcl_sha256, 16)
    except ValueError as error:
        raise ValueError("bundle SHA-256 is not hexadecimal") from error
    if open3d_sha256 != pcl_sha256:
        raise ValueError("Open3D and PCL canonical bundle SHA-256 differ")
